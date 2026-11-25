# Implementation plan for the no-IRF S8 continuation and the next model

The immediate run continues the trained S8 Transformer with contraction-onset loss weighting disabled. It uses the existing causal-scattering architecture. The next full model will use a frozen FHR forecaster and a separate learned UA residual. Standalone CouplingIRF v2 is the timing benchmark for that research.

## Run the S8 continuation

Open [trainer.py](trainer.py) on the production machine and press Run with the project's Python environment. Its default config is [run_s8_no_irf.yaml](configs/run_s8_no_irf.yaml). The equivalent module launch from the repository root is:

```bash
python -m teb_vae.lag_attn_transformer_cfs.trainer
```

The inherited device list selects seven GPUs, and Lightning starts the workers from a single process. To launch with an external process per GPU, use:

```bash
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_transformer_cfs.trainer \
    --config teb_vae/lag_attn_transformer_cfs/configs/run_s8_no_irf.yaml
```

The config uses the original production path recorded in the epoch-369 checkpoint, the 2026-11-18 healthy-only training and validation shards, and the statistics used for that checkpoint. These paths require the production mounts. If the checkpoint has moved, update `model_config.core_model_checkpoint` to the trained epoch-369 file at its new location. The launch starts a fresh optimizer from the saved weights. It runs 20 epochs with a 500-update warm-up to $3.0\times10^{-5}$, the learning rate in the saved optimizer at epoch 369. The original per-device batch size and gradient accumulation are inherited.

The sole intended change to the training objective is removing contraction-onset weighting. The config sets `event_weight_floor: null` and disables the weighting detector's `event_weight_ua_valid` input, as required by preflight. The posterior event gate and its penalty remain active. The source path is `lag_attn`, both IRF switches are off, and the IRF and full wrong-UA loss weights are zero. The run keeps the sampled base decode, 38-lag persistence skip, Student-$t$ likelihood with $\nu=8$, AR(1) innovation cap of $0.95$, source-null KL penalty, and constant $\beta=1$. The data retain their canonical stored clock and the existing UA shift of $-20\,\mathrm{s}$.

### Read the run against S8

1. Re-evaluate the original epoch-369 checkpoint and candidate checkpoints through the same evaluator with the same held-out recordings, stats, masks, forecast horizons, and scoring rules. Pass shards and stats explicitly to the lag-analysis evaluator. The older evaluation override has a different cohort and retention settings from its sibling. Use the same checkpoint selection rule across comparisons. Recompute the corrected mean gap from saved weights for the S8 baseline.
2. Track FHR-only and full NLL, scattering and phase mean error by horizon, and uncertainty calibration. Use `val/pred_gap_mean` for the paired latent-mean comparison. Both branches are decoded at their means in that readout. Use a shared Monte Carlo scoring protocol for probabilistic comparisons.
3. On batches that support a cross-recording permutation, compare the in-training shuffle with the matched full forecast. If any DDP rank lacks a valid cross-GUID permutation, every rank skips the shuffle metric for that batch. For checkpoint decisions about UA specificity, use the [matched donor evaluator](eval/lag_analysis/core.py) with shared noise and report paired own-UA and wrong-UA gains on common scored cells. Report selection rules, donor coverage, and recording-level uncertainty.
4. Treat this as a short continuation ablation. A decline in own-UA gain alongside improving FHR-only forecast quality needs the matched donor comparison to establish whether UA specificity improved. The fixed-profile control from the [epoch-369 audit](../../tmp/lag_analysis_369/REPORT.md) retained 95–100% of the forecast gain, so attention lag remains an alignment diagnostic.

## Next full model

### Forecasting and predictive coupling

Train a probabilistic FHR-only causal TCN on stitched 4-second signals. Start with FHR level, within-step variability, and validity. Match targets, masks, and horizon across the persistence, ridge, and TCN baselines. Check forecast error by horizon and uncertainty calibration before freezing the base. The S7 audit's FHR-only long ridge scored $0.557$ scattering MSE. Its $0.543$ result included UA. Those comparisons were on scattering targets, so measure the stitched FHR-level baselines directly.

Freeze the FHR base and its likelihood parameters. Train a small causal UA temporal convolution for a zero-initialized mean correction. Use UA history available at the forecast anchor. Validate with aligned UA, a cross-GUID donor matched on labour stage and coverage, time-shifted own UA, and a value-null control retaining the validity flags. Require positive own-UA gain and a positive own-minus-wrong margin on common scored cells. Set an equivalence tolerance around zero for wrong-UA gain on development data. If a wrong-UA training penalty is helpful, penalize departure from the base predictive distribution under donor UA. Add a variance correction or a latent only if it improves held-out calibration or predictive gain under those controls.

Call the held-out log-score difference *predictive gain*. Report forecast with $U_{\le t}$ and retrospective explain with UA available through the target separately. An information-theoretic interpretation requires a specified conditioning set and calibrated predictive densities. The observed gain ratio across these modes does not identify a fraction of physiological coupling.

### Timing and outcome research

Treat response timing as a separate benchmark and model output. Start with the stitched CouplingIRF v2 acceptance results, then compare a contraction-aligned head with an explicit response probability, signed response shape, and delay. Define the landmark for the delay before training. For a horizon indexed by $\tau\in\{1,\ldots,H\}$, the stored-grid delay from source step $t_k$ to forecast target $t+\tau$ is $d=t+\tau-t_k$. Test timing on strong, weak, intermittent, depth-zero, and shifted-UA plant conditions. Report error jointly with detection coverage and evaluate agreement with held-out real events. Separate features available by the anchor from completed-event features in retrospective analysis. Keep the stored-clock convention in repository comparisons and check recorded-clock availability separately before interpreting a result as a live forecast.

Standalone v2 recovered well-expressed planted delays with median errors of $0.97$–$2.67\,\mathrm{s}$ at $20\,\mathrm{bpm}$ depth. The intermittent condition detected $15.4\%$ of eligible anchors and the split-half gain reliability missed its acceptance threshold. Use these as baseline measurements. If revisiting the IRF, test centering over valid convolution taps, define minimum prequential fit support, and recalibrate on development recordings. Test each change against v2 before combining it with adaptive context or a nonlinear event head.

For outcome modelling, start from the [latent pilot's cohort and baseline protocol](LATENT_CLASS_FINETUNING_PILOT.md). Lock data and model choices on development recordings, document pretraining and checkpoint-selection exposure, and use a fresh confirmation cohort for new claims informed by existing test reports. Group by patient when identifiers are available and bootstrap at the recording or patient level. Treat the reported clinical subgroup differences as descriptive.
