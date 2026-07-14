# Raw-signal VAE-TEB v4 (`SeqVaeRawV4`)

`SeqVaeRawV4` is `SeqVaeLagAttnV3` with its fixed scattering/phase adapters replaced by two learned,
strictly-causal, anti-aliased, cumulative-normalised multiscale **front ends** over the raw 4 Hz
FHR/UP signals, and its feature-domain decoder replaced by a **raw future-FHR forecaster**. Every v3
scientific-cleanliness guarantee (G0–G11) is inherited unchanged; the only new obligation is that the
causal-normalisation guarantee (G0) is pushed down into the front end.

- **Design / math:** [`../vae-teb-raw-model.md`](../vae-teb-raw-model.md)
- **Spec + sprint roadmap:** [`../vae-teb-raw-v4-spec-and-sprints.md`](../vae-teb-raw-v4-spec-and-sprints.md)
- Interactive explainer: `../docs/vae-teb-v3.html` (the v3 core this forks).

## Entry points

**Train** (single-phase, end-to-end; DDP on the prod box, single GPU for smoke):

```bash
python -m model.vae_teb_prediction.model.model_raw.trainer_raw_v4 --config \
  model/vae_teb_prediction/model/model_raw/config_raw_v4.yaml
```

**Evaluate** a checkpoint (raw metrics + plots, G10 calibration, G11 CMI, latent/KL/attention/TE,
causal-TE validation):

```bash
python -m model.vae_teb_prediction.model.model_raw.testing.run_tests \
  --checkpoint <path.ckpt> --config \
  model/vae_teb_prediction/model/model_raw/testing/config_raw_v4_testing.yaml \
  --output <out_dir>
```

**Test** (unit + causality + geometry; add the slow train-then-assert known-answer harnesses):

```bash
pytest model/vae_teb_prediction/model/model_raw/testing            # full suite
pytest model/vae_teb_prediction/model/model_raw/testing -m "not slow"   # fast tier only
```

## Configs

| File | Purpose |
|---|---|
| `config_raw_v4.yaml` | Authoritative production training config (raw_len 5280 → T=300, DDP). |
| `config_raw_v4_smoke.yaml` | Single-GPU memory/OOM tuning harness (epochs 2). |
| `testing/config_raw_v4_testing.yaml` | Eval/analysis pipeline config (empty datasets; `run_tests`). |
| `testing/config_tiny.yaml` | Tiny build-from-config target for the trainer unit tests. |
| `configs/ablation_*.yaml` | The §16 ablation suite — each is `config_raw_v4.yaml` with a single flag flipped (`single_stride16`, `no_antialias`, `plain_activation`, `linear_head`, `disable_source`). |

## Layout

- `vae_teb_raw_v4.py` — `SeqVaeRawV4(SeqVaeLagAttnV3)` + raw future decoders.
- `raw_frontend.py` — `CausalRawFrontend` (causal multiscale front end, G0-in-front-end).
- `geometry.py` — config-driven raw/low-rate geometry + the crop-offset identities.
- `raw_targets.py` / `raw_masks.py` / `raw_losses.py` — crop-aligned target, masks, single-phase loss.
- `trainer_raw_v4.py` — `SeqVaeRawV4Pl` + `GraphModelVaeTebRawV4Trainer` (single-phase, full metrics).
- `testing/` — raw-domain metrics/plots + the domain-agnostic latent/KL/attention/TE/calibration/CMI
  suite + causality, ablation, and synthetic-lag known-answer harnesses.
