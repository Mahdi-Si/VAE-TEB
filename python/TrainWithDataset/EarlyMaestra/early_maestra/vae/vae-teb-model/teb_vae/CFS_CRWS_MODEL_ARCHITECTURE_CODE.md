# Model architectures and full network code — lag_attn_cfs / lag_attn_transformer_cfs / lag_attn_crws / lag_attn_transformer_crws

Companion to `teb_vae/CFS_CRWS_MODELS_REFERENCE.md` (data pipeline, configs, objective, training, eval). This file holds only the **network architecture**: every `nn.Module`, mixin and constructor the four models are built from, in dependency order, with the exact code, plus the module trees and forward shape traces of the four models instantiated from their shipped `configs/default.yaml`. Written 2026-09-05.

**CFS review status, 2026-09-05:** the [independent critique](CFS_SCATTERING_PHASE_FORECAST_CRITIQUE.md) identifies mathematical and interpretive limitations; the [implementation task list](CFS_SCATTERING_PHASE_FIX_TASKS.md) tracks their fixes. This document still captures the current executable architecture, including its legacy phase widths and clock geometry. The new notes below correct interpretation; they do not represent implemented network changes. Numbered source excerpts, captured code blocks, module trees, and current parameter counts are retained. Regenerate them from executable source only when the corresponding implementation changes (CFS-15).

How to read the code blocks:

- Each block is the source file with **docstrings and full-line comments removed**; every remaining line is verbatim. The prefix `NNNN|` is the line number in the original file, so `path:NNNN` pointers from the companion document land here directly. Blank-line runs are collapsed.
- Omitted on purpose (metric/readout code, not architecture; see the companion §1.8): `FeatureForecastTarget._resolved_forecast_gaps/_forecast_gaps_from_mask` (`teb_vae/lag_attn_fs/nets/feature_target.py:200-324`), `CausalFeatureForecastTarget._resolve_target_warm_frac/_rank_tertiles/_resolve_warm_tertiles/_resolve_novelty_tertiles/_resolve_block_warm_steps/_gap_by_kept_channel/_anchors_per_sample/_source_lag_warmth/_resolved_forecast_gaps` (`causal_feature_target.py:305-432, 853-1102`), `CausalWarmupInputs._resolve_warmup_readout_constants` (`causal_inputs.py:436-568`), `CausalRawInputs._resolve_warmup_readout_constants` (`causal_raw_inputs.py:313-381`). Also not reproduced: `teb_vae/lag_attn_rws/nets/losses.py`, `raw_masks.py`, `raw_targets.py` (objective; formulas in the companion), `controls.py` (source-null / permutation controls), `lag_report.py`.
- Trees and shape traces were produced by instantiating each class with the kwargs the trainer derives from `default.yaml` (signature sweep + `warmup_model_kwargs(resolve_warmup_budget(...))`), with the shard paths pointed at the committed fixture `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5` (same filter-bank attributes as production shards). Totals equal the numbers in each package's `DESIGN.md`.

---

## 0. Architecture at a glance

### 0.1 One pipeline, two axes

All four models share one forward (`CausalWarmupInputs.forward`, §12) and differ along two axes: the **encoder** (conv-LSTM `CausalConvLstmEncoder` vs conv-Transformer `CausalConvTransformerEncoder`) and the **target** (98 kept one-sided feature channels vs 16 raw FHR samples per horizon step).

```
 y_st (B,T,36) ─┐                                         u_stream (B,T,51)
 y_ph (B,T,66) ─┴─ cat ─► target (B,T,102)                     │
                             │                                  │
                      target_gate: ChannelGate           source_gate: ChannelGate
                      gather keep_index + delay d_c      gather keep_index + delay d_c
                      (B,T,C_keep_y)                     (B,T,C_keep_u)
                             │                                  │
                      target_adapter:                    source_adapter:
                      AvailabilityInputAdapter           AvailabilityInputAdapter
                      mask m_t,c=1[t>=W'_c+d_c]          (B,T,128)
                      (B,T,128)                                 │
                             │                           encode_source_kv  (lag_kv_source)
                      target_encoder                       'conv_stem' -> source_kv_stem  (shipped)
                      conv-LSTM | conv-Transformer         'encoder'   -> source_encoder
                      h_y (B,T,128)                        'adapter'   -> identity
                             │                           h_u (B,T,128) = keys AND values
                             │                                  │
              prior_head: FullLatentPriorHead(h_y, clock)       │   clock = encode_source_kv(gate(0))
              mu_p, logvar_p, raw_logvar_p (B,T,d_z=64)         │   (B=1, detached, eval-mode)
                             │                                  │
                query_proj(mu_p) (B,T,128) ──► lag_attn: LagCrossAttention(q, h_u, mask (T,L=91))
                                                 alpha (B,T,4,91), attended_heads (B,T,4,32)
                             │                                  │
              posterior_head: PosteriorHead(h_y, attended_heads, mu_p, raw_logvar_p)
              mu_q = mu_p + 3 tanh(./3), logvar_q (independent head)   (B,T,64)
                             │
              _reparameterize_shared: eps ~ N(0,I); z_q = mu_q + sigma_q eps; z_p = mu_p (base_decode=mean)
                             │
              anchors A(phi,S) = {F+phi+kS < ceiling}  -> gather z[:, anchors]  (B,A,64)
                             │
              decoder: BaselineFutureDecoder(HorizonDecoderCore)   invoked twice, same weights
                 proj ResidualMLP 64->256 ; core: +horizon_embedding(30,256), 4 dilated Conv1d(k3, d=1,2,4,8)+GN+GELU
                 with per-block FiLM from z, 2 horizon self-attention blocks, LayerNorm(feat+skip)
                 mean_head Linear(256->C_out), logvar_head Linear(256->C_out) -> smooth_bound(-5,3)
                 (+ persistence_weight(30,C_out) * y_anchor on feature cells)
                 -> mu_base/logvar_base from z_p, mu_full/logvar_full from z_q : (B,A,30,C_out)
                             │
              kld_tensor(mu_p,logvar_p,mu_q,logvar_q) (B,T,64) -> te_analysis -> kld_per_t (B,T),
              source_kl_lag_map (B,T,91) = sum_m KL_head_m * alpha_m, kld_per_t_per_head (B,T,4)
```

$C_{\mathrm{out}}=98$ (cfs cells, `FeatureForecastTarget._default_decoder_out_channels` = kept target channels) or $16$ (crws cells, `raw_per_step`). Every stream tensor is $(B,T,\cdot)$ with $T=300$; forecast tensors are $(B,A_{\max},H=30,C_{\mathrm{out}})$.

### 0.2 The four classes

| model | class (bases, in MRO order) | file | target encoder | source K/V module (shipped `lag_kv_source: conv_stem`) | gates target/source (kept, max shift) | decoder out | params (trainable) |
|---|---|---|---|---|---|---|---|
| cfs | `SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws)` | `teb_vae/lag_attn_cfs/nets/model.py:40` | `CausalConvLstmEncoder` kernels (3,7,11,15,15) dil (1,2,4,8,16) + 2-layer LSTM | `CausalConvStem` kernels (3,5,11,15,15) dil (1,2,4,8,16), RF 387 steps | 98/102 (85) / 39/51 (60) | 98 | 4,655,987 (4,639,475) |
| trf-cfs | `SeqVaeLagAttnTrfCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws)` | `teb_vae/lag_attn_transformer_cfs/nets/model.py:56` | `CausalConvTransformerEncoder` 2 gated conv blocks (5,9)/(1,2) + 6 `CausalTransformerBlock` full causal prefix | `GatedCausalConvStem` 2 blocks, RF 21 steps | 98/102 (85) / 39/51 (60) | 98 | 4,284,556 (4,268,044) |
| crws | `SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws)` | `teb_vae/lag_attn_crws/nets/model.py:53` | as cfs | as cfs | 38/102 (6) / 17/51 (6) | 16 | 4,589,907 (4,573,395) |
| trf-crws | `SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws)` | `teb_vae/lag_attn_transformer_crws/nets/model.py:68` | as trf-cfs | as trf-cfs | 38/102 (6) / 17/51 (6) | 16 | 4,218,476 (4,201,964) |

The 16,512 frozen parameters in every model are `lag_attn.W_o` (weight 128×128 + bias), unused because the posterior consumes per-head summaries. Non-shipped arm `lag_kv_source: encoder` builds a full source encoder instead of the stem (conv-LSTM 1,312,231 params; conv-Transformer 888,960 with 3 windowed blocks) — those are the "off-state" totals 5,146,334 / 5,054,992 / 5,081,146 / 4,989,804 in the design records.

### 0.3 Top-level attribute → class → source

| attribute | class | defined in | notes |
|---|---|---|---|
| `target_gate`, `source_gate` | `ChannelGate` → `ChannelDelay` | `teb_vae/lag_attn/nets/delays.py` | built iff keep_index or delays given (`_build_channel_gate`) |
| `target_adapter`, `source_adapter` | `AvailabilityInputAdapter` | `teb_vae/lag_attn/nets/encoders.py:95` | causal cells build with delays $W'_c+d_c$ (`CausalWarmupInputs._build_adapter`) |
| `target_encoder` | `CausalConvLstmEncoder` / `CausalConvTransformerEncoder` | `lag_attn/nets/encoders.py:383` / `lag_attn_transformer_rws/nets/encoders.py:82` | |
| `source_encoder` \| `source_kv_stem` \| (none) | `CausalConvLstmEncoder` or `CausalConvStem` / `CausalConvTransformerEncoder` or `GatedCausalConvStem` | same files | exactly one built per `lag_kv_source` |
| `prior_head` | `FullLatentPriorHead` | `teb_vae/lag_attn_rws/nets/heads.py:48` | `clock_proj` Linear(128→128, no bias, zero-init) present iff `prior_availability_input` |
| `query_proj` | `nn.Linear(d_z→d_model)` | `lag_attn_rws/nets/model.py:583` | |
| `lag_attn` | `LagCrossAttention` | `teb_vae/lag_attn/nets/attention.py:65` | `lag_embeddings (91,4,32)`, `lag_score_bias (4,91)` (alibi_decay; flat when slope scale 0) |
| `posterior_head` | `PosteriorHead(head_structured=True)` | `teb_vae/lag_attn/nets/heads.py:139` | 4 per-head fusions `ResidualMLP(160→32)`, `delta_mu_head` 4×Linear(32→16), `logvar_post_head` 4×Linear(32→16) |
| `te_analysis` | `TEAnalysisHead` | `lag_attn/nets/heads.py:415` | parameter-free |
| `horizon_core` | `HorizonDecoderCore` | `teb_vae/lag_attn/nets/decoders.py:236` | shared object; also registered inside `decoder.core` (counted twice in per-child table, once in total) |
| `decoder` | `BaselineFutureDecoder` | `lag_attn/nets/decoders.py:388` | `persistence_weight (30,98)` on feature cells |
| buffers | `future_index (270,30,16)`, `horizon_weight (30,)`, `target_channel_weight (98,)`, `warm_tertile_id`, `novelty_tertile_id`, `source_block_warm_st/ph (300,)`, gate `keep_index`/`delay_steps`, adapter `availability (300,C)`, `start_indicator (300,1)`, RoPE `cos/sin_table (300,16)`, source `attn_mask (300,300)` | | all non-persistent |

### 0.4 Construction and initialisation order (`SeqVaeLagAttnRws.__init__`, mirrored by `SeqVaeLagAttnTrfRws.__init__`)

1. Mixin pre-step (causal cells): `_set_causal_inputs` (+ `_set_channel_weights`, `_set_target_novelty` on feature cells).
2. Validate widths/heads/lag/`d_z % num_heads`; `TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)`.
3. Buffers `future_index`, `horizon_weight`; gates; adapters (mixin override announces $W'_c+d_c$); target encoder; source K/V module per `lag_kv_source`; `prior_head` (clock width `_prior_clock_dim()` = `d_model` on causal cells); `query_proj`; `lag_attn` (dropout 0); `posterior_head` (head-structured); `te_analysis`; `horizon_core` (`film_per_block=True`); `decoder` (width from `_default_decoder_out_channels()`, dropout 0, persistence per `_check_persistence_target`).
4. `causal_norm` → `causalize_norms` on target encoder and the source K/V body (conv-LSTM cells only). Freeze `lag_attn.W_o`.
5. `initialization(self)` (xavier Linear/Conv, orthogonal LSTM, forget bias 1); transformer cells then `init_depthwise_` (σ = 1/√k on depthwise convs).
6. Re-zero after generic init: `_zero_init_delta_heads` (delta_mu heads zero; independent logvar head weight 0, bias $\log(5/3)$), `prior_head.zero_init_clock()`, `_zero_init_film_generators`.
7. Init policy (shipped): `_reinit_horizon_embedding` (std 0.8), `_calibrate_output_heads` (mean_head ×0.02, logvar bias $\log(5/3)$, logvar weight ×0.1), `_calibrate_prior_scale` (zero skip + final weight, bias $\log(5/3)$), `_set_a_head_gain` (LayerNorm gain 2.0 on attended summary).
8. Mixin post-step: `_validate_causal_geometry` (stride vs span, floor), `_register_channel_weights` (feature cells), readout constants.

### 0.5 CFS interpretation limits and planned changes

These notes apply to the CFS review. Shared code also serves CRWS and historical models; future CFS migrations must preserve those consumers' explicit legacy contracts.

| Current architecture fact | Interpretation / implementation task |
|---|---|
| CFS tensors use 36 scattering + 66 phase target channels, with 98 kept decoder outputs | The current phase extraction includes a discontinuous noninteger harmonic family. The corrected integer-phase schema will change dimensions and channel identity, but is not implemented in this snapshot. CFS-01–03 must version pair/operator metadata, rebuild stats, and enforce checkpoint compatibility before updating these counts. |
| `ChannelGate` gathers/delays channels; the largest CFS target shift is 85 steps | This withholds 340 s of recent fast-channel trajectories from the encoder. Global alignment is optional for causal inputs, and the energy-centroid delay convention is not an exact physical timestamp. Test unshifted/grouped inputs and fresh/slow paths (CFS-05/09/12/13). |
| Persistence is gathered before the target gate | On the current `physical` and `stored` clocks, it receives the current kept target values. It is a target-only decoder path outside the latent. Thus the encoder loses recent trajectories under alignment, but the entire model does not lose current levels. Both branches receive the same persistence input. |
| CFS target gather uses $t+1+h+s_c$ | `physical` is approximate target delay compensation, not just an axis label: the present 30-step output includes labels through 460 s after the anchor and has 51 dense anchors. `input` can select already-available labels. Input reference and scored-target reference must be distinguished (CFS-05/09). |
| Availability adapters use $W'_c+d_c$ | They encode the configured 95%-energy initialization policy, not exact independence from padding or support-aware signal quality. Captured log text such as `honest by` is the implementation's wording, not an independently validated guarantee. Continuous history and quality propagation are pending (CFS-06/07). |
| CFS source values have RF 387 steps for the convolution stem, 21 steps for the transformer stem | A lag index selects a history summary, not necessarily a point event. The longer stem exceeds the input segment length. Neither receptive-field size nor an attention peak independently identifies physiological delay; test local values and raw-domain interventions (CFS-11). |
| `TEAnalysisHead` distributes head KL using attention weights | The name is retained from source. The tensor is an attention-weighted latent-divergence attribution, not measured transfer entropy. The source-conditioned branch does not observe future target labels; `posterior` is its architectural name. CFS-10/11 define the reporting and validation work. |
| Prior uses its mean; the full branch samples its latent | Equal latent distributions do not give equal training-path forecasts. Matched Monte Carlo evaluation samples both branches with common random numbers; use that policy for predictive source-gain claims (CFS-10). |

**User-specified timeline convention:** the dataset-created UA/UP timeline is canonical. The accepted 20-second dataset-creation adjustment is ignored in all downstream interpretation and compensation; it is not a causality criticism or a remediation target. CFS-04 tests other repair/resampling operations on that canonical timeline, excluding the accepted adjustment itself. CFS-05 brings downstream lag reporting into the same zero-correction convention. The captured network code remains unchanged.

The proposed primary candidate keeps one-sided integer-phase features, envelope-plus-phasor leg alignment, unshifted input channels, stored-clock targets, and target-only persistence. The proposed fresh/slow architecture is an experiment, not present in the code below. Both need matched held-out comparisons; no source-code block in this document should be hand-edited to portray them as already implemented.

---

## 1. Shipped module trees and shape traces

### 1.1 `SeqVaeLagAttnCfs` — `teb_vae/lag_attn_cfs/configs/default.yaml`

kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 102, "c_u": 51, "use_up_st": true, "lstm_layers": 2, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "causal_norm": true, "anchor_stride": 5, "lag_floor": 0, "target_weight_st": 1.0, "target_weight_ph": 0.1, "horizon_weight_halflife_steps": 15.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "persistence_residual": true, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "encoder_extra_dilations": [8, 16], "encoder_extra_kernel": 15, "target_keep_index": "<98 entries, min 0 max 101>", "target_warmup_steps": "<98 entries, min 0 max 134>", "source_keep_index": "<39 entries, min 0 max 44>", "source_warmup_steps": "<39 entries, min 0 max 92>", "target_novelty_frac": "<102 entries, min 0.0024794680066406727 max 1.0>", "target_forecast_shift": "<98 entries, min 0 max 85>", "target_align_delays": "<98 entries, min 0 max 85>", "source_align_delays": "<39 entries, min 0 max 60>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, target reference 402.1604 s, source reference 288.2672 s, inter-stream offset -113.8932 s, forecast clock physical (shift 0..85 steps, ceiling -85)): target: fhr_st 32/36, fhr_ph 66/66; 98/102 channels, warm-up 0-134 steps, shift 0-85 steps, honest by 0-134; source: up_st 30/36, up_ph 9/15; 39/51 channels, warm-up 0-92 steps, shift 0-60 steps, honest by 0-92
params total=4,655,987 trainable=4,639,475 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                     0
  source_gate                     0
  target_adapter             92,672
  source_adapter             77,568
  target_encoder          1,344,999
  source_kv_stem            804,352
  prior_head                124,650
  query_proj                  8,320
  lag_attn                   78,572
  posterior_head            116,484
  te_analysis                     0
  horizon_core            1,849,346
  decoder                 2,008,370
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('warm_tertile_id', (98,)), ('novelty_tertile_id', (98,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_channel_weight', (98,)), ('target_gate.keep_index', (98,)), ('target_gate.delay.delay_steps', (98,)), ('source_gate.keep_index', (39,)), ('source_gate.delay.delay_steps', (39,)), ('target_adapter.availability', (300, 98)), ('target_adapter.start_indicator', (300, 1)), ('source_adapter.availability', (300, 39)), ('source_adapter.start_indicator', (300, 1))]
gates (out_channels, max_delay): target=(98, 85) source=(39, 60)
decoder_out_channels=98 anchor_ceiling=185 anchor_stride=5 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: (0, 85)
```
SeqVaeLagAttnCfs(
  (target_gate): ChannelGate(
    102 -> 98, max_delay=85
    (delay): ChannelDelay(num_channels=98, max_delay=85)
  )
  (source_gate): ChannelGate(
    51 -> 39, max_delay=60
    (delay): ChannelDelay(num_channels=39, max_delay=60)
  )
  (target_adapter): AvailabilityInputAdapter(
    98 -> 128, max_delay=134, availability_terms=['W_m', 'e_start']
    (linear): Linear(in_features=98, out_features=128, bias=True)
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (act): GELU(approximate='none')
    (drop): Dropout(p=0.1, inplace=False)
    (res_mlp): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=128, out_features=128, bias=True)
        (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=128, out_features=128, bias=True)
      )
      (skip_proj): Identity()
    )
    (mask_proj): Linear(in_features=98, out_features=128, bias=False)
  )
  (source_adapter): AvailabilityInputAdapter(
    39 -> 128, max_delay=92, availability_terms=['W_m', 'e_start']
    (linear): Linear(in_features=39, out_features=128, bias=True)
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (act): GELU(approximate='none')
    (drop): Dropout(p=0.1, inplace=False)
    (res_mlp): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=128, out_features=128, bias=True)
        (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=128, out_features=128, bias=True)
      )
      (skip_proj): Identity()
    )
    (mask_proj): Linear(in_features=39, out_features=128, bias=False)
  )
  (target_encoder): CausalConvLstmEncoder(
    (front_mlp): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=128, out_features=128, bias=True)
        (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=128, out_features=128, bias=True)
      )
      (skip_proj): Identity()
    )
    (convs): ModuleList(
      (0): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(3,), stride=(1,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(7,), stride=(1,), dilation=(2,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(11,), stride=(1,), dilation=(4,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(15,), stride=(1,), dilation=(8,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(15,), stride=(1,), dilation=(16,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (conv_out_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (lstm): LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.1)
    (lstm_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (fusion): ResidualMLP(
      (input_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=256, out_features=215, bias=True)
        (1): LayerNorm((215,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=215, out_features=181, bias=True)
        (5): LayerNorm((181,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=181, out_features=152, bias=True)
        (9): LayerNorm((152,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=152, out_features=128, bias=True)
      )
      (skip_proj): Linear(in_features=256, out_features=128, bias=True)
    )
    (output_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (source_kv_stem): CausalConvStem(
    d_model=128, kernels=(3, 5, 11, 15, 15), dilations=(1, 2, 4, 8, 16), receptive_field=387 steps
    (convs): ModuleList(
      (0): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(3,), stride=(1,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(5,), stride=(1,), dilation=(2,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(11,), stride=(1,), dilation=(4,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(15,), stride=(1,), dilation=(8,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): CausalMultiChannelConvBlock(
        (pre_norm): CausalGroupNorm(8, 128, eps=1e-05)
        (conv): Conv1d(128, 128, kernel_size=(15,), stride=(1,), dilation=(16,), bias=False)
        (act_fn): GELU(approximate='none')
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (output_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (prior_head): FullLatentPriorHead(
    (mu_input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (logvar_input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (clock_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (clock_proj): Linear(in_features=128, out_features=128, bias=False)
    (mu_prior_head): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=111, bias=True)
        (1): LayerNorm((111,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=111, out_features=97, bias=True)
        (5): LayerNorm((97,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=97, out_features=84, bias=True)
        (9): LayerNorm((84,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=84, out_features=74, bias=True)
        (13): LayerNorm((74,), eps=1e-05, elementwise_affine=True)
        (14): GELU(approximate='none')
        (15): Dropout(p=0.1, inplace=False)
        (16): Linear(in_features=74, out_features=64, bias=True)
      )
      (skip_proj): Linear(in_features=128, out_features=64, bias=True)
    )
    (logvar_prior_head): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=111, bias=True)
        (1): LayerNorm((111,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=111, out_features=97, bias=True)
        (5): LayerNorm((97,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=97, out_features=84, bias=True)
        (9): LayerNorm((84,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=84, out_features=74, bias=True)
        (13): LayerNorm((74,), eps=1e-05, elementwise_affine=True)
        (14): GELU(approximate='none')
        (15): Dropout(p=0.1, inplace=False)
        (16): Linear(in_features=74, out_features=64, bias=True)
      )
      (skip_proj): Linear(in_features=128, out_features=64, bias=True)
    )
  )
  (query_proj): Linear(in_features=64, out_features=128, bias=True)
  (lag_attn): LagCrossAttention(
    (q_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (kv_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (W_q): Linear(in_features=128, out_features=128, bias=True)
    (W_k): Linear(in_features=128, out_features=128, bias=True)
    (W_v): Linear(in_features=128, out_features=128, bias=True)
    (W_o): Linear(in_features=128, out_features=128, bias=True)
    (attn_dropout): Dropout(p=0.0, inplace=False)
  )
  (posterior_head): PosteriorHead(
    (h_y_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (a_dropout): Dropout(p=0.0, inplace=False)
    (a_head_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
    (fusion): ModuleList(
      (0-3): 4 x ResidualMLP(
        (input_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
        (body): Sequential(
          (0): Linear(in_features=160, out_features=94, bias=True)
          (1): LayerNorm((94,), eps=1e-05, elementwise_affine=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=94, out_features=55, bias=True)
          (5): LayerNorm((55,), eps=1e-05, elementwise_affine=True)
          (6): GELU(approximate='none')
          (7): Dropout(p=0.1, inplace=False)
          (8): Linear(in_features=55, out_features=32, bias=True)
          (9): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        )
        (skip_proj): Linear(in_features=160, out_features=32, bias=True)
        (final_act): GELU(approximate='none')
      )
    )
    (delta_mu_head): ModuleList(
      (0-3): 4 x Linear(in_features=32, out_features=16, bias=True)
    )
    (logvar_post_head): ModuleList(
      (0-3): 4 x Linear(in_features=32, out_features=16, bias=True)
    )
  )
  (te_analysis): TEAnalysisHead()
  (horizon_core): HorizonDecoderCore(
    (refine): _HorizonRefine(
      (blocks): ModuleList(
        (0): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
        (1): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
        (2): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
        (3): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
      )
      (film): ModuleList(
        (0-3): 4 x Linear(in_features=256, out_features=512, bias=True)
      )
    )
    (attention): ModuleList(
      (0-1): 2 x _HorizonSelfAttention(
        (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (q_proj): Linear(in_features=256, out_features=256, bias=False)
        (k_proj): Linear(in_features=256, out_features=256, bias=False)
        (v_proj): Linear(in_features=256, out_features=256, bias=False)
        (out_proj): Linear(in_features=256, out_features=256, bias=False)
      )
    )
    (out_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): BaselineFutureDecoder(
    (core): HorizonDecoderCore(
      (refine): _HorizonRefine(
        (blocks): ModuleList(
          (0): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
          (1): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
          (2): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
          (3): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
        )
        (film): ModuleList(
          (0-3): 4 x Linear(in_features=256, out_features=512, bias=True)
        )
      )
      (attention): ModuleList(
        (0-1): 2 x _HorizonSelfAttention(
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (q_proj): Linear(in_features=256, out_features=256, bias=False)
          (k_proj): Linear(in_features=256, out_features=256, bias=False)
          (v_proj): Linear(in_features=256, out_features=256, bias=False)
          (out_proj): Linear(in_features=256, out_features=256, bias=False)
        )
      )
      (out_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (proj): ResidualMLP(
      (input_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=64, out_features=91, bias=True)
        (1): LayerNorm((91,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Linear(in_features=91, out_features=128, bias=True)
        (4): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (5): GELU(approximate='none')
        (6): Linear(in_features=128, out_features=181, bias=True)
        (7): LayerNorm((181,), eps=1e-05, elementwise_affine=True)
        (8): GELU(approximate='none')
        (9): Linear(in_features=181, out_features=256, bias=True)
        (10): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (skip_proj): Linear(in_features=64, out_features=256, bias=True)
      (final_act): GELU(approximate='none')
    )
    (mean_head): Linear(in_features=256, out_features=98, bias=True)
    (logvar_head): Linear(in_features=256, out_features=98, bias=True)
  )
)
```

### 1.2 `SeqVaeLagAttnTrfCfs` — `teb_vae/lag_attn_transformer_cfs/configs/default.yaml`

kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 102, "c_u": 51, "use_up_st": true, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "encoder_conv_kernels": [5, 9], "encoder_conv_dilations": [1, 2], "encoder_num_heads": 4, "encoder_d_ff": 512, "target_attention_blocks": 6, "source_attention_blocks": 3, "source_attention_window": 16, "anchor_stride": 5, "lag_floor": 0, "target_weight_st": 1.0, "target_weight_ph": 0.1, "horizon_weight_halflife_steps": 15.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "persistence_residual": true, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "target_keep_index": "<98 entries, min 0 max 101>", "target_warmup_steps": "<98 entries, min 0 max 134>", "source_keep_index": "<39 entries, min 0 max 44>", "source_warmup_steps": "<39 entries, min 0 max 92>", "target_novelty_frac": "<102 entries, min 0.0024794680066406727 max 1.0>", "target_forecast_shift": "<98 entries, min 0 max 85>", "target_align_delays": "<98 entries, min 0 max 85>", "source_align_delays": "<39 entries, min 0 max 60>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, target reference 402.1604 s, source reference 288.2672 s, inter-stream offset -113.8932 s, forecast clock physical (shift 0..85 steps, ceiling -85)): target: fhr_st 32/36, fhr_ph 66/66; 98/102 channels, warm-up 0-134 steps, shift 0-85 steps, honest by 0-134; source: up_st 30/36, up_ph 9/15; 39/51 channels, warm-up 0-92 steps, shift 0-60 steps, honest by 0-92
params total=4,284,556 trainable=4,268,044 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                     0
  source_gate                     0
  target_adapter             92,672
  source_adapter             77,568
  target_encoder          1,676,928
  source_kv_stem            100,992
  prior_head                124,650
  query_proj                  8,320
  lag_attn                   78,572
  posterior_head            116,484
  te_analysis                     0
  horizon_core            1,849,346
  decoder                 2,008,370
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('warm_tertile_id', (98,)), ('novelty_tertile_id', (98,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_channel_weight', (98,)), ('target_gate.keep_index', (98,)), ('target_gate.delay.delay_steps', (98,)), ('source_gate.keep_index', (39,)), ('source_gate.delay.delay_steps', (39,)), ('target_adapter.availability', (300, 98)), ('target_adapter.start_indicator', (300, 1)), ('source_adapter.availability', (300, 39)), ('source_adapter.start_indicator', (300, 1))]
gates (out_channels, max_delay): target=(98, 85) source=(39, 60)
decoder_out_channels=98 anchor_ceiling=185 anchor_stride=5 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: (0, 85)
```
SeqVaeLagAttnTrfCfs(
  (target_gate): ChannelGate(
    102 -> 98, max_delay=85
    (delay): ChannelDelay(num_channels=98, max_delay=85)
  )
  (source_gate): ChannelGate(
    51 -> 39, max_delay=60
    (delay): ChannelDelay(num_channels=39, max_delay=60)
  )
  (target_adapter): AvailabilityInputAdapter(
    98 -> 128, max_delay=134, availability_terms=['W_m', 'e_start']
    (linear): Linear(in_features=98, out_features=128, bias=True)
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (act): GELU(approximate='none')
    (drop): Dropout(p=0.1, inplace=False)
    (res_mlp): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=128, out_features=128, bias=True)
        (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=128, out_features=128, bias=True)
      )
      (skip_proj): Identity()
    )
    (mask_proj): Linear(in_features=98, out_features=128, bias=False)
  )
  (source_adapter): AvailabilityInputAdapter(
    39 -> 128, max_delay=92, availability_terms=['W_m', 'e_start']
    (linear): Linear(in_features=39, out_features=128, bias=True)
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (act): GELU(approximate='none')
    (drop): Dropout(p=0.1, inplace=False)
    (res_mlp): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=128, out_features=128, bias=True)
        (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=128, out_features=128, bias=True)
      )
      (skip_proj): Identity()
    )
    (mask_proj): Linear(in_features=39, out_features=128, bias=False)
  )
  (target_encoder): CausalConvTransformerEncoder(
    d_model=128, conv_blocks=2, attention_blocks=6, context=full causal prefix, receptive_field=unbounded
    (conv_blocks): ModuleList(
      (0): GatedCausalConvBlock(
        (norm_in): RMSNorm(128, eps=1e-05)
        (proj_in): Linear(in_features=128, out_features=256, bias=False)
        (conv): CausalDepthwiseConv1d(
          128, kernel_size=5, dilation=1, left_padding=4
          (conv): Conv1d(128, 128, kernel_size=(5,), stride=(1,), groups=128, bias=False)
        )
        (norm_conv): RMSNorm(128, eps=1e-05)
        (proj_out): Linear(in_features=128, out_features=128, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
        (layer_scale): LayerScale(128, init=0.01)
      )
      (1): GatedCausalConvBlock(
        (norm_in): RMSNorm(128, eps=1e-05)
        (proj_in): Linear(in_features=128, out_features=256, bias=False)
        (conv): CausalDepthwiseConv1d(
          128, kernel_size=9, dilation=2, left_padding=16
          (conv): Conv1d(128, 128, kernel_size=(9,), stride=(1,), dilation=(2,), groups=128, bias=False)
        )
        (norm_conv): RMSNorm(128, eps=1e-05)
        (proj_out): Linear(in_features=128, out_features=128, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
        (layer_scale): LayerScale(128, init=0.01)
      )
    )
    (attention_blocks): ModuleList(
      (0-5): 6 x CausalTransformerBlock(
        (attn): CausalSelfAttention(
          d_model=128, num_heads=4, context=full causal prefix
          (norm): RMSNorm(128, eps=1e-05)
          (q_proj): Linear(in_features=128, out_features=128, bias=False)
          (k_proj): Linear(in_features=128, out_features=128, bias=False)
          (v_proj): Linear(in_features=128, out_features=128, bias=False)
          (out_proj): Linear(in_features=128, out_features=128, bias=False)
          (rope): RotaryPositionEncoding(d_head=32, max_seq_len=300, base=10000.0)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (attn_scale): LayerScale(128, init=0.01)
        (ffn_norm): RMSNorm(128, eps=1e-05)
        (ffn): SwiGLUFeedForward(
          (gate_proj): Linear(in_features=128, out_features=512, bias=False)
          (value_proj): Linear(in_features=128, out_features=512, bias=False)
          (out_proj): Linear(in_features=512, out_features=128, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (ffn_scale): LayerScale(128, init=0.01)
      )
    )
    (output_norm): RMSNorm(128, eps=1e-05)
  )
  (source_kv_stem): GatedCausalConvStem(
    d_model=128, conv_blocks=2, receptive_field=21 steps
    (conv_blocks): ModuleList(
      (0): GatedCausalConvBlock(
        (norm_in): RMSNorm(128, eps=1e-05)
        (proj_in): Linear(in_features=128, out_features=256, bias=False)
        (conv): CausalDepthwiseConv1d(
          128, kernel_size=5, dilation=1, left_padding=4
          (conv): Conv1d(128, 128, kernel_size=(5,), stride=(1,), groups=128, bias=False)
        )
        (norm_conv): RMSNorm(128, eps=1e-05)
        (proj_out): Linear(in_features=128, out_features=128, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
        (layer_scale): LayerScale(128, init=0.01)
      )
      (1): GatedCausalConvBlock(
        (norm_in): RMSNorm(128, eps=1e-05)
        (proj_in): Linear(in_features=128, out_features=256, bias=False)
        (conv): CausalDepthwiseConv1d(
          128, kernel_size=9, dilation=2, left_padding=16
          (conv): Conv1d(128, 128, kernel_size=(9,), stride=(1,), dilation=(2,), groups=128, bias=False)
        )
        (norm_conv): RMSNorm(128, eps=1e-05)
        (proj_out): Linear(in_features=128, out_features=128, bias=False)
        (dropout): Dropout(p=0.1, inplace=False)
        (layer_scale): LayerScale(128, init=0.01)
      )
    )
    (output_norm): RMSNorm(128, eps=1e-05)
  )
  (prior_head): FullLatentPriorHead(
    (mu_input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (logvar_input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (clock_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (clock_proj): Linear(in_features=128, out_features=128, bias=False)
    (mu_prior_head): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=111, bias=True)
        (1): LayerNorm((111,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=111, out_features=97, bias=True)
        (5): LayerNorm((97,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=97, out_features=84, bias=True)
        (9): LayerNorm((84,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=84, out_features=74, bias=True)
        (13): LayerNorm((74,), eps=1e-05, elementwise_affine=True)
        (14): GELU(approximate='none')
        (15): Dropout(p=0.1, inplace=False)
        (16): Linear(in_features=74, out_features=64, bias=True)
      )
      (skip_proj): Linear(in_features=128, out_features=64, bias=True)
    )
    (logvar_prior_head): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=111, bias=True)
        (1): LayerNorm((111,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=111, out_features=97, bias=True)
        (5): LayerNorm((97,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=97, out_features=84, bias=True)
        (9): LayerNorm((84,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=84, out_features=74, bias=True)
        (13): LayerNorm((74,), eps=1e-05, elementwise_affine=True)
        (14): GELU(approximate='none')
        (15): Dropout(p=0.1, inplace=False)
        (16): Linear(in_features=74, out_features=64, bias=True)
      )
      (skip_proj): Linear(in_features=128, out_features=64, bias=True)
    )
  )
  (query_proj): Linear(in_features=64, out_features=128, bias=True)
  (lag_attn): LagCrossAttention(
    (q_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (kv_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (W_q): Linear(in_features=128, out_features=128, bias=True)
    (W_k): Linear(in_features=128, out_features=128, bias=True)
    (W_v): Linear(in_features=128, out_features=128, bias=True)
    (W_o): Linear(in_features=128, out_features=128, bias=True)
    (attn_dropout): Dropout(p=0.0, inplace=False)
  )
  (posterior_head): PosteriorHead(
    (h_y_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (a_dropout): Dropout(p=0.0, inplace=False)
    (a_head_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
    (fusion): ModuleList(
      (0-3): 4 x ResidualMLP(
        (input_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
        (body): Sequential(
          (0): Linear(in_features=160, out_features=94, bias=True)
          (1): LayerNorm((94,), eps=1e-05, elementwise_affine=True)
          (2): GELU(approximate='none')
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=94, out_features=55, bias=True)
          (5): LayerNorm((55,), eps=1e-05, elementwise_affine=True)
          (6): GELU(approximate='none')
          (7): Dropout(p=0.1, inplace=False)
          (8): Linear(in_features=55, out_features=32, bias=True)
          (9): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        )
        (skip_proj): Linear(in_features=160, out_features=32, bias=True)
        (final_act): GELU(approximate='none')
      )
    )
    (delta_mu_head): ModuleList(
      (0-3): 4 x Linear(in_features=32, out_features=16, bias=True)
    )
    (logvar_post_head): ModuleList(
      (0-3): 4 x Linear(in_features=32, out_features=16, bias=True)
    )
  )
  (te_analysis): TEAnalysisHead()
  (horizon_core): HorizonDecoderCore(
    (refine): _HorizonRefine(
      (blocks): ModuleList(
        (0): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
        (1): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
        (2): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
        (3): ModuleDict(
          (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
          (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
        )
      )
      (film): ModuleList(
        (0-3): 4 x Linear(in_features=256, out_features=512, bias=True)
      )
    )
    (attention): ModuleList(
      (0-1): 2 x _HorizonSelfAttention(
        (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (q_proj): Linear(in_features=256, out_features=256, bias=False)
        (k_proj): Linear(in_features=256, out_features=256, bias=False)
        (v_proj): Linear(in_features=256, out_features=256, bias=False)
        (out_proj): Linear(in_features=256, out_features=256, bias=False)
      )
    )
    (out_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): BaselineFutureDecoder(
    (core): HorizonDecoderCore(
      (refine): _HorizonRefine(
        (blocks): ModuleList(
          (0): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
          (1): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
          (2): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(4,), dilation=(4,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
          (3): ModuleDict(
            (conv): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(8,), dilation=(8,))
            (norm): GroupNorm(8, 256, eps=1e-05, affine=True)
          )
        )
        (film): ModuleList(
          (0-3): 4 x Linear(in_features=256, out_features=512, bias=True)
        )
      )
      (attention): ModuleList(
        (0-1): 2 x _HorizonSelfAttention(
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (q_proj): Linear(in_features=256, out_features=256, bias=False)
          (k_proj): Linear(in_features=256, out_features=256, bias=False)
          (v_proj): Linear(in_features=256, out_features=256, bias=False)
          (out_proj): Linear(in_features=256, out_features=256, bias=False)
        )
      )
      (out_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (proj): ResidualMLP(
      (input_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=64, out_features=91, bias=True)
        (1): LayerNorm((91,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Linear(in_features=91, out_features=128, bias=True)
        (4): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (5): GELU(approximate='none')
        (6): Linear(in_features=128, out_features=181, bias=True)
        (7): LayerNorm((181,), eps=1e-05, elementwise_affine=True)
        (8): GELU(approximate='none')
        (9): Linear(in_features=181, out_features=256, bias=True)
        (10): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (skip_proj): Linear(in_features=64, out_features=256, bias=True)
      (final_act): GELU(approximate='none')
    )
    (mean_head): Linear(in_features=256, out_features=98, bias=True)
    (logvar_head): Linear(in_features=256, out_features=98, bias=True)
  )
)
```

### 1.3 `SeqVaeLagAttnCrws` — `teb_vae/lag_attn_crws/configs/default.yaml`

Header (kwargs, budget, params, buffers, gates):
kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 102, "c_u": 51, "use_up_st": true, "lstm_layers": 2, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "causal_norm": true, "anchor_stride": 30, "lag_floor": 0, "horizon_weight_halflife_steps": 15.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "encoder_extra_dilations": [8, 16], "encoder_extra_kernel": 15, "target_keep_index": "<38 entries, min 0 max 56>", "target_warmup_steps": "<38 entries, min 0 max 1>", "source_keep_index": "<17 entries, min 0 max 16>", "source_warmup_steps": "<17 entries, min 0 max 1>", "target_align_delays": "<38 entries, min 0 max 6>", "source_align_delays": "<17 entries, min 0 max 6>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, reference 42.2066 s): target: fhr_st 17/36, fhr_ph 21/66; 38/102 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6; source: up_st 17/36, up_ph 0/15; 17/51 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6
params total=4,589,907 trainable=4,573,395 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                     0
  source_gate                     0
  target_adapter             77,312
  source_adapter             71,936
  target_encoder          1,344,999
  source_kv_stem            804,352
  prior_head                124,650
  query_proj                  8,320
  lag_attn                   78,572
  posterior_head            116,484
  te_analysis                     0
  horizon_core            1,849,346
  decoder                 1,963,282
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_gate.keep_index', (38,)), ('target_gate.delay.delay_steps', (38,)), ('source_gate.keep_index', (17,)), ('source_gate.delay.delay_steps', (17,)), ('target_adapter.availability', (300, 38)), ('target_adapter.start_indicator', (300, 1)), ('source_adapter.availability', (300, 17)), ('source_adapter.start_indicator', (300, 1))]
gates (out_channels, max_delay): target=(38, 6) source=(17, 6)
decoder_out_channels=16 anchor_ceiling=270 anchor_stride=30 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: None

Module tree = the cfs tree above except these lines (unified diff, cfs -> crws):
```
@@ -1,16 +1,16 @@
 ```
-SeqVaeLagAttnCfs(
+SeqVaeLagAttnCrws(
   (target_gate): ChannelGate(
-    102 -> 98, max_delay=85
-    (delay): ChannelDelay(num_channels=98, max_delay=85)
+    102 -> 38, max_delay=6
+    (delay): ChannelDelay(num_channels=38, max_delay=6)
   )
   (source_gate): ChannelGate(
-    51 -> 39, max_delay=60
-    (delay): ChannelDelay(num_channels=39, max_delay=60)
+    51 -> 17, max_delay=6
+    (delay): ChannelDelay(num_channels=17, max_delay=6)
   )
   (target_adapter): AvailabilityInputAdapter(
-    98 -> 128, max_delay=134, availability_terms=['W_m', 'e_start']
-    (linear): Linear(in_features=98, out_features=128, bias=True)
+    38 -> 128, max_delay=6, availability_terms=['W_m', 'e_start']
+    (linear): Linear(in_features=38, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
     (act): GELU(approximate='none')
     (drop): Dropout(p=0.1, inplace=False)
@@ -33,11 +33,11 @@
       )
       (skip_proj): Identity()
     )
-    (mask_proj): Linear(in_features=98, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=38, out_features=128, bias=False)
   )
   (source_adapter): AvailabilityInputAdapter(
-    39 -> 128, max_delay=92, availability_terms=['W_m', 'e_start']
-    (linear): Linear(in_features=39, out_features=128, bias=True)
+    17 -> 128, max_delay=6, availability_terms=['W_m', 'e_start']
+    (linear): Linear(in_features=17, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
     (act): GELU(approximate='none')
     (drop): Dropout(p=0.1, inplace=False)
@@ -60,7 +60,7 @@
       )
       (skip_proj): Identity()
     )
-    (mask_proj): Linear(in_features=39, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=17, out_features=128, bias=False)
   )
   (target_encoder): CausalConvLstmEncoder(
     (front_mlp): ResidualMLP(
@@ -356,8 +356,8 @@
       (skip_proj): Linear(in_features=64, out_features=256, bias=True)
       (final_act): GELU(approximate='none')
     )
-    (mean_head): Linear(in_features=256, out_features=98, bias=True)
-    (logvar_head): Linear(in_features=256, out_features=98, bias=True)
+    (mean_head): Linear(in_features=256, out_features=16, bias=True)
+    (logvar_head): Linear(in_features=256, out_features=16, bias=True)
   )
 )
 ```
```

### 1.4 `SeqVaeLagAttnTrfCrws` — `teb_vae/lag_attn_transformer_crws/configs/default.yaml`

Header (kwargs, budget, params, buffers, gates):
kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 102, "c_u": 51, "use_up_st": true, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "encoder_conv_kernels": [5, 9], "encoder_conv_dilations": [1, 2], "encoder_num_heads": 4, "encoder_d_ff": 512, "target_attention_blocks": 6, "source_attention_blocks": 3, "source_attention_window": 16, "anchor_stride": 30, "lag_floor": 0, "horizon_weight_halflife_steps": 15.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "target_keep_index": "<38 entries, min 0 max 56>", "target_warmup_steps": "<38 entries, min 0 max 1>", "source_keep_index": "<17 entries, min 0 max 16>", "source_warmup_steps": "<17 entries, min 0 max 1>", "target_align_delays": "<38 entries, min 0 max 6>", "source_align_delays": "<17 entries, min 0 max 6>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, reference 42.2066 s): target: fhr_st 17/36, fhr_ph 21/66; 38/102 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6; source: up_st 17/36, up_ph 0/15; 17/51 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6
params total=4,218,476 trainable=4,201,964 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                     0
  source_gate                     0
  target_adapter             77,312
  source_adapter             71,936
  target_encoder          1,676,928
  source_kv_stem            100,992
  prior_head                124,650
  query_proj                  8,320
  lag_attn                   78,572
  posterior_head            116,484
  te_analysis                     0
  horizon_core            1,849,346
  decoder                 1,963,282
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_gate.keep_index', (38,)), ('target_gate.delay.delay_steps', (38,)), ('source_gate.keep_index', (17,)), ('source_gate.delay.delay_steps', (17,)), ('target_adapter.availability', (300, 38)), ('target_adapter.start_indicator', (300, 1)), ('source_adapter.availability', (300, 17)), ('source_adapter.start_indicator', (300, 1))]
gates (out_channels, max_delay): target=(38, 6) source=(17, 6)
decoder_out_channels=16 anchor_ceiling=270 anchor_stride=30 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: None

Module tree = the tcfs tree above except these lines (unified diff, tcfs -> tcrws):
```
@@ -1,16 +1,16 @@
 ```
-SeqVaeLagAttnTrfCfs(
+SeqVaeLagAttnTrfCrws(
   (target_gate): ChannelGate(
-    102 -> 98, max_delay=85
-    (delay): ChannelDelay(num_channels=98, max_delay=85)
+    102 -> 38, max_delay=6
+    (delay): ChannelDelay(num_channels=38, max_delay=6)
   )
   (source_gate): ChannelGate(
-    51 -> 39, max_delay=60
-    (delay): ChannelDelay(num_channels=39, max_delay=60)
+    51 -> 17, max_delay=6
+    (delay): ChannelDelay(num_channels=17, max_delay=6)
   )
   (target_adapter): AvailabilityInputAdapter(
-    98 -> 128, max_delay=134, availability_terms=['W_m', 'e_start']
-    (linear): Linear(in_features=98, out_features=128, bias=True)
+    38 -> 128, max_delay=6, availability_terms=['W_m', 'e_start']
+    (linear): Linear(in_features=38, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
     (act): GELU(approximate='none')
     (drop): Dropout(p=0.1, inplace=False)
@@ -33,11 +33,11 @@
       )
       (skip_proj): Identity()
     )
-    (mask_proj): Linear(in_features=98, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=38, out_features=128, bias=False)
   )
   (source_adapter): AvailabilityInputAdapter(
-    39 -> 128, max_delay=92, availability_terms=['W_m', 'e_start']
-    (linear): Linear(in_features=39, out_features=128, bias=True)
+    17 -> 128, max_delay=6, availability_terms=['W_m', 'e_start']
+    (linear): Linear(in_features=17, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
     (act): GELU(approximate='none')
     (drop): Dropout(p=0.1, inplace=False)
@@ -60,7 +60,7 @@
       )
       (skip_proj): Identity()
     )
-    (mask_proj): Linear(in_features=39, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=17, out_features=128, bias=False)
   )
   (target_encoder): CausalConvTransformerEncoder(
     d_model=128, conv_blocks=2, attention_blocks=6, context=full causal prefix, receptive_field=unbounded
@@ -327,8 +327,8 @@
       (skip_proj): Linear(in_features=64, out_features=256, bias=True)
       (final_act): GELU(approximate='none')
     )
-    (mean_head): Linear(in_features=256, out_features=98, bias=True)
-    (logvar_head): Linear(in_features=256, out_features=98, bias=True)
+    (mean_head): Linear(in_features=256, out_features=16, bias=True)
+    (logvar_head): Linear(in_features=256, out_features=16, bias=True)
   )
 )
 ```
```

### 1.5 Forward shape traces (random inputs, `eval()` mode, forward hooks on top-level children)

The three source-side hook lines at batch 1 are the prior clock: `_prior_clock` re-runs `source_gate → source_adapter → source_kv_stem` on a zero stream of batch 1. The decoder appears twice (prior latent, then posterior latent) with the same weights.

### SeqVaeLagAttnCfs: forward at B=2, T=300, anchor_stride=5, anchor_phase=0
```
  target_gate          in=[(2, 300, 102)] out=(2, 300, 98)
  source_gate          in=[(2, 300, 51)] out=(2, 300, 39)
  target_adapter       in=[(2, 300, 98)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 39)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 51)] out=(1, 300, 39)
  source_adapter       in=[(1, 300, 39)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 11, 64)] out=[(2, 11, 30, 98), (2, 11, 30, 98)]
  decoder              in=[(2, 11, 64)] out=[(2, 11, 30, 98), (2, 11, 30, 98)]
outputs:
  mu_prior                 (2, 300, 64) float32
  logvar_prior             (2, 300, 64) float32
  raw_logvar_prior         (2, 300, 64) float32
  mu_post                  (2, 300, 64) float32
  logvar_post              (2, 300, 64) float32
  z_prior                  (2, 300, 64) float32
  z_post                   (2, 300, 64) float32
  target_state             (2, 300, 128) float32
  source_state             (2, 300, 128) float32
  attended_source_heads    (2, 300, 4, 32) float32
  attn_weights             (2, 300, 4, 91) float32
  mu_base                  (2, 11, 30, 98) float32
  logvar_base              (2, 11, 30, 98) float32
  mu_full                  (2, 11, 30, 98) float32
  logvar_full              (2, 11, 30, 98) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 11) int64
  anchor_valid             (2, 11) bool
  persistence              (2, 11, 98) float32
```
### SeqVaeLagAttnCfs (dense val/test geometry): forward at B=2, T=300, anchor_stride=1, anchor_phase=0
```
  target_gate          in=[(2, 300, 102)] out=(2, 300, 98)
  source_gate          in=[(2, 300, 51)] out=(2, 300, 39)
  target_adapter       in=[(2, 300, 98)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 39)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 51)] out=(1, 300, 39)
  source_adapter       in=[(1, 300, 39)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 51, 64)] out=[(2, 51, 30, 98), (2, 51, 30, 98)]
  decoder              in=[(2, 51, 64)] out=[(2, 51, 30, 98), (2, 51, 30, 98)]
outputs:
  mu_prior                 (2, 300, 64) float32
  logvar_prior             (2, 300, 64) float32
  raw_logvar_prior         (2, 300, 64) float32
  mu_post                  (2, 300, 64) float32
  logvar_post              (2, 300, 64) float32
  z_prior                  (2, 300, 64) float32
  z_post                   (2, 300, 64) float32
  target_state             (2, 300, 128) float32
  source_state             (2, 300, 128) float32
  attended_source_heads    (2, 300, 4, 32) float32
  attn_weights             (2, 300, 4, 91) float32
  mu_base                  (2, 51, 30, 98) float32
  logvar_base              (2, 51, 30, 98) float32
  mu_full                  (2, 51, 30, 98) float32
  logvar_full              (2, 51, 30, 98) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 51) int64
  anchor_valid             (2, 51) bool
  persistence              (2, 51, 98) float32
```
### SeqVaeLagAttnTrfCfs: forward at B=2, T=300, anchor_stride=5, anchor_phase=0
```
  target_gate          in=[(2, 300, 102)] out=(2, 300, 98)
  source_gate          in=[(2, 300, 51)] out=(2, 300, 39)
  target_adapter       in=[(2, 300, 98)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 39)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 51)] out=(1, 300, 39)
  source_adapter       in=[(1, 300, 39)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 11, 64)] out=[(2, 11, 30, 98), (2, 11, 30, 98)]
  decoder              in=[(2, 11, 64)] out=[(2, 11, 30, 98), (2, 11, 30, 98)]
outputs:
  mu_prior                 (2, 300, 64) float32
  logvar_prior             (2, 300, 64) float32
  raw_logvar_prior         (2, 300, 64) float32
  mu_post                  (2, 300, 64) float32
  logvar_post              (2, 300, 64) float32
  z_prior                  (2, 300, 64) float32
  z_post                   (2, 300, 64) float32
  target_state             (2, 300, 128) float32
  source_state             (2, 300, 128) float32
  attended_source_heads    (2, 300, 4, 32) float32
  attn_weights             (2, 300, 4, 91) float32
  mu_base                  (2, 11, 30, 98) float32
  logvar_base              (2, 11, 30, 98) float32
  mu_full                  (2, 11, 30, 98) float32
  logvar_full              (2, 11, 30, 98) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 11) int64
  anchor_valid             (2, 11) bool
  persistence              (2, 11, 98) float32
```
### SeqVaeLagAttnCrws: forward at B=2, T=300, anchor_stride=30, anchor_phase=0
```
  target_gate          in=[(2, 300, 102)] out=(2, 300, 38)
  source_gate          in=[(2, 300, 51)] out=(2, 300, 17)
  target_adapter       in=[(2, 300, 38)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 17)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 51)] out=(1, 300, 17)
  source_adapter       in=[(1, 300, 17)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 5, 64)] out=[(2, 5, 30, 16), (2, 5, 30, 16)]
  decoder              in=[(2, 5, 64)] out=[(2, 5, 30, 16), (2, 5, 30, 16)]
outputs:
  mu_prior                 (2, 300, 64) float32
  logvar_prior             (2, 300, 64) float32
  raw_logvar_prior         (2, 300, 64) float32
  mu_post                  (2, 300, 64) float32
  logvar_post              (2, 300, 64) float32
  z_prior                  (2, 300, 64) float32
  z_post                   (2, 300, 64) float32
  target_state             (2, 300, 128) float32
  source_state             (2, 300, 128) float32
  attended_source_heads    (2, 300, 4, 32) float32
  attn_weights             (2, 300, 4, 91) float32
  mu_base                  (2, 5, 30, 16) float32
  logvar_base              (2, 5, 30, 16) float32
  mu_full                  (2, 5, 30, 16) float32
  logvar_full              (2, 5, 30, 16) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 5) int64
  anchor_valid             (2, 5) bool
```
### SeqVaeLagAttnTrfCrws: forward at B=2, T=300, anchor_stride=30, anchor_phase=0
```
  target_gate          in=[(2, 300, 102)] out=(2, 300, 38)
  source_gate          in=[(2, 300, 51)] out=(2, 300, 17)
  target_adapter       in=[(2, 300, 38)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 17)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 51)] out=(1, 300, 17)
  source_adapter       in=[(1, 300, 17)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 5, 64)] out=[(2, 5, 30, 16), (2, 5, 30, 16)]
  decoder              in=[(2, 5, 64)] out=[(2, 5, 30, 16), (2, 5, 30, 16)]
outputs:
  mu_prior                 (2, 300, 64) float32
  logvar_prior             (2, 300, 64) float32
  raw_logvar_prior         (2, 300, 64) float32
  mu_post                  (2, 300, 64) float32
  logvar_post              (2, 300, 64) float32
  z_prior                  (2, 300, 64) float32
  z_post                   (2, 300, 64) float32
  target_state             (2, 300, 128) float32
  source_state             (2, 300, 128) float32
  attended_source_heads    (2, 300, 4, 32) float32
  attn_weights             (2, 300, 4, 91) float32
  mu_base                  (2, 5, 30, 16) float32
  logvar_base              (2, 5, 30, 16) float32
  mu_full                  (2, 5, 30, 16) float32
  logvar_full              (2, 5, 30, 16) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 5) int64
  anchor_valid             (2, 5) bool
```

---

## 2. Shared primitives — `teb_vae/lag_attn/nets/blocks.py`

`geometric_schedule` (hidden widths on a geometric ramp, e.g. `geometric_schedule(128, 64, 4) = (111, 97, 84, 74, 64)`), `initialization`, `CausalMultiChannelConvBlock` (pre-norm GroupNorm → act → left-pad → Conv1d → dropout → +residual; `(B,C,L)`), `ResidualMLP` (LayerNorm input → Linear/LN/GELU/Dropout stack → + skip from normalised input → optional final GELU; per-timestep), `CausalGroupNorm` (GroupNorm statistics per timestep, state-dict compatible with `nn.GroupNorm`), `causalize_norms`, `smooth_bound(r, lo, hi) = lo + (hi-lo)·sigmoid(r)`.

`teb_vae/lag_attn/nets/blocks.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   12| from __future__ import annotations
   13| 
   14| import copy
   15| from typing import Callable, Optional, Tuple, cast
   16| 
   17| import torch
   18| import torch.nn.functional as F
   19| from torch import nn
   20| 
   22| def geometric_schedule(
   23|     input_size: int,
   24|     output_size: int,
   25|     n_hidden: int,
   26|     *,
   27|     round_fn: Callable[[float], float] = round,
   28| ) -> Tuple[int, ...]:
   45|     steps = n_hidden + 1
   46|     ratio = (output_size / input_size) ** (1 / steps)
   47| 
   48|     sizes = [input_size]
   49|     current_ratio = ratio
   50|     for _ in range(n_hidden):
   51|         sizes.append(int(round_fn(input_size * current_ratio)))
   52|         current_ratio *= ratio
   53|     sizes.append(output_size)
   54| 
   55|     return tuple(sizes[1:])
   56| 
   58| def initialization(model: nn.Module) -> None:
   72|     for _, module in model.named_modules():
   73|         if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
   74|             nn.init.xavier_uniform_(module.weight)
   75|             if module.bias is not None:
   76|                 nn.init.zeros_(module.bias)
   77|         elif isinstance(module, nn.LSTM):
   78|             for param_name, param in module.named_parameters():
   79|                 if "weight_ih" in param_name or "weight_hh" in param_name:
   80|                     nn.init.orthogonal_(param)
   81|                 elif "bias" in param_name:
   82|                     nn.init.zeros_(param)
   83|                     if "bias_hh" in param_name:
   84|                         hidden_size = module.hidden_size
   85|                         param.data[hidden_size : 2 * hidden_size].fill_(1.0)
   86|         elif isinstance(module, nn.LayerNorm):
   87|             nn.init.ones_(module.weight)
   88|             nn.init.zeros_(module.bias)
   89| 
   91| class CausalMultiChannelConvBlock(nn.Module):
  102| 
  103|     def __init__(
  104|         self,
  105|         in_channels: int = 1,
  106|         out_channels: int = 1,
  107|         groups: int = 1,
  108|         filter_size: int = 3,
  109|         activation: type[nn.Module] = nn.ReLU,
  110|         dilation: int = 1,
  111|         stride: int = 1,
  112|         bias: bool = False,
  113|         dropout: float = 0.2,
  114|         norm_groups: Optional[int] = None,
  115|     ) -> None:
  134|         super().__init__()
  135| 
  136|         self.left_padding = (filter_size - 1) * dilation
  139|         pre_norm_groups = min(8, in_channels) if norm_groups is None else int(norm_groups)
  140|         self.pre_norm = nn.GroupNorm(num_groups=pre_norm_groups, num_channels=in_channels)
  141|         self.conv = nn.Conv1d(
  142|             in_channels,
  143|             out_channels,
  144|             kernel_size=filter_size,
  145|             groups=groups,
  146|             bias=bias,
  147|             padding=0,
  148|             dilation=dilation,
  149|             stride=stride,
  150|         )
  151|         self.act_fn = activation()
  152| 
  154|         self.residual_proj = None
  155|         if in_channels != out_channels:
  156|             self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
  157| 
  158|         self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
  159| 
  160|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  169|         residual = x
  170|         x = self.pre_norm(x)
  171|         x = self.act_fn(x)
  172|         if self.left_padding > 0:
  173|             x = F.pad(x, (self.left_padding, 0))
  174|         output = self.conv(x)
  175| 
  176|         if self.dropout is not None:
  177|             output = self.dropout(output)
  178| 
  179|         if self.residual_proj is not None:
  180|             residual = self.residual_proj(residual)
  181| 
  182|         return output + residual
  183| 
  185| class ResidualMLP(nn.Module):
  195| 
  196|     def __init__(
  197|         self,
  198|         input_dim: int,
  199|         hidden_dims: Tuple[int, ...] = (72, 68, 64),
  200|         final_activation: bool = True,
  201|         activation: type[nn.Module] = nn.GELU,
  202|         use_skip_connection: bool = True,
  203|         use_input_layer_norm: bool = True,
  204|         dropout: float = 0.1,
  205|     ) -> None:
  217|         super().__init__()
  218|         self.final_activation = final_activation
  219|         self.use_skip_connection = use_skip_connection
  220|         self.dropout = dropout
  221|         self.input_norm = nn.LayerNorm(input_dim) if use_input_layer_norm else nn.Identity()
  222|         self.activation_factory = self._build_activation_factory(activation)
  223| 
  224|         layers: list[nn.Module] = []
  225|         dims = [input_dim, *hidden_dims]
  226|         for index in range(len(hidden_dims)):
  227|             is_final_layer = index == len(hidden_dims) - 1
  228|             layers.append(nn.Linear(dims[index], dims[index + 1]))
  229|             if not is_final_layer or final_activation:
  230|                 layers.append(nn.LayerNorm(dims[index + 1]))
  231|             if not is_final_layer:
  232|                 layers.append(self.activation_factory())
  233|                 if dropout > 0.0:
  234|                     layers.append(nn.Dropout(dropout))
  235|         self.body = nn.Sequential(*layers)
  236| 
  237|         final_dim = hidden_dims[-1]
  238|         if self.use_skip_connection:
  241|             self.skip_proj = (
  242|                 nn.Linear(input_dim, final_dim) if input_dim != final_dim else nn.Identity()
  243|             )
  244|         else:
  245|             self.skip_proj = None
  246| 
  247|         self.final_act = self.activation_factory() if final_activation else None
  248| 
  249|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  261|         x_norm = self.input_norm(x)
  262|         y = self.body(x_norm)
  263|         if self.use_skip_connection and self.skip_proj is not None:
  264|             y = y + self.skip_proj(x_norm)
  265|         if self.final_activation and self.final_act is not None:
  266|             y = self.final_act(y)
  267|         return y
  268| 
  269|     @staticmethod
  270|     def _build_activation_factory(activation) -> Callable[[], nn.Module]:
  286|         if isinstance(activation, nn.Module):
  287|             return lambda: copy.deepcopy(activation)
  288|         if isinstance(activation, type) and issubclass(activation, nn.Module):
  289|             return activation
  290|         if callable(activation):
  292|             return cast(Callable[[], nn.Module], activation)
  293|         raise TypeError(
  294|             "activation must be an nn.Module instance, nn.Module subclass, or callable "
  295|             "returning an nn.Module."
  296|         )
  297| 
  299| class CausalGroupNorm(nn.Module):
  324| 
  325|     def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
  336|         super().__init__()
  337|         if num_channels % num_groups != 0:
  338|             raise ValueError(
  339|                 f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
  340|             )
  341|         self.num_groups = int(num_groups)
  342|         self.num_channels = int(num_channels)
  343|         self.eps = float(eps)
  344|         self.weight = nn.Parameter(torch.ones(num_channels))
  345|         self.bias = nn.Parameter(torch.zeros(num_channels))
  346| 
  347|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  356|         batch, channels, _ = x.shape
  357|         grouped = x.view(batch, self.num_groups, channels // self.num_groups, -1)
  358|         mean = grouped.mean(dim=2, keepdim=True)
  359|         var = grouped.var(dim=2, unbiased=False, keepdim=True)
  360|         normed = ((grouped - mean) / torch.sqrt(var + self.eps)).view(batch, channels, -1)
  361|         return normed * self.weight[None, :, None] + self.bias[None, :, None]
  362| 
  363|     def extra_repr(self) -> str:
  365|         return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"
  366| 
  368| def causalize_norms(module: nn.Module) -> int:
  380|     replaced = 0
  381|     for name, child in module.named_children():
  382|         if isinstance(child, nn.GroupNorm):
  383|             causal = CausalGroupNorm(child.num_groups, child.num_channels, child.eps)
  384|             if child.affine:
  385|                 with torch.no_grad():
  386|                     causal.weight.copy_(child.weight)
  387|                     causal.bias.copy_(child.bias)
  388|                 causal.to(child.weight.device)
  389|             setattr(module, name, causal)
  390|             replaced += 1
  391|         else:
  392|             replaced += causalize_norms(child)
  393|     return replaced
  394| 
  396| def smooth_bound(r: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
  416|     return lo + (hi - lo) * torch.sigmoid(r)
  417| 
  419| def validate_choice(value: str, choices: Tuple[str, ...], name: str) -> str:
  433|     if value not in choices:
  434|         raise ValueError(f"{name} must be one of {choices}, got {value!r}")
  435|     return value
```

## 3. Channel gate — `teb_vae/lag_attn/nets/delays.py`

`ChannelGate.forward(x (B,T,C_declared)) = ChannelDelay(index_select(x, keep_index))`; `ChannelDelay` reads channel $c$ at step $t-\delta_c$ (clamped) and zeroes steps $t<\delta_c$. In the causal cells `delays` = alignment shifts $d_c$ (companion §1.4), zeros when unaligned.

`teb_vae/lag_attn/nets/delays.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   32| from __future__ import annotations
   33| 
   34| from typing import Optional, Sequence
   35| 
   36| import torch
   37| from torch import nn
   38| 
   40| class ChannelDelay(nn.Module):
   48| 
   50|     delay_steps: torch.Tensor
   51| 
   52|     def __init__(self, *, num_channels: int, delays: Sequence[int]) -> None:
   66|         super().__init__()
   67| 
   68|         num_channels = int(num_channels)
   69|         if num_channels < 1:
   70|             raise ValueError(f"num_channels must be >= 1, got {num_channels}")
   71| 
   72|         delay_values = [int(value) for value in delays]
   73|         if len(delay_values) != num_channels:
   74|             raise ValueError(
   75|                 f"delays has {len(delay_values)} entries but num_channels is {num_channels}. "
   76|                 f"The delay vector is positional -- one entry per surviving channel, in channel "
   77|                 f"order -- so a length mismatch would delay the wrong channels with no other "
   78|                 f"failure signal."
   79|             )
   80|         negative = [(index, value) for index, value in enumerate(delay_values) if value < 0]
   81|         if negative:
   82|             raise ValueError(
   83|                 f"delays must be >= 0; got negative entries at "
   84|                 f"{negative}. A negative delay reads a channel from its own future, which is "
   85|                 f"the leak this module exists to remove."
   86|             )
   87| 
   88|         self.num_channels = num_channels
   90|         self.max_delay = max(delay_values)
   91| 
   96|         self.register_buffer(
   97|             "delay_steps", torch.tensor(delay_values, dtype=torch.long), persistent=False
   98|         )
   99| 
  100|     def extra_repr(self) -> str:
  102|         return f"num_channels={self.num_channels}, max_delay={self.max_delay}"
  103| 
  104|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  117|         if x.dim() != 3 or x.shape[-1] != self.num_channels:
  118|             raise ValueError(
  119|                 f"expected a (B, T, {self.num_channels}) stream, got {tuple(x.shape)}"
  120|             )
  121| 
  124|         steps = torch.arange(x.shape[1], device=x.device).unsqueeze(-1)
  125|         source_index = steps - self.delay_steps.unsqueeze(0)
  126|         available = source_index >= 0
  127| 
  130|         gathered = x.gather(
  131|             1, source_index.clamp_min(0).unsqueeze(0).expand(x.shape[0], -1, -1)
  132|         )
  133|         return gathered * available.to(x.dtype)
  134| 
  136| class ChannelGate(nn.Module):
  147| 
  149|     keep_index: torch.Tensor
  150| 
  151|     def __init__(
  152|         self,
  153|         *,
  154|         declared_width: int,
  155|         keep_index: Optional[Sequence[int]] = None,
  156|         delays: Optional[Sequence[int]] = None,
  157|     ) -> None:
  179|         super().__init__()
  180| 
  181|         width = int(declared_width)
  182|         indices = list(range(width)) if keep_index is None else [int(i) for i in keep_index]
  183|         if not indices:
  184|             raise ValueError(
  185|                 "keep_index is empty: the model would train to completion having never read "
  186|                 "this stream."
  187|             )
  188|         outside = [index for index in indices if index < 0 or index >= width]
  189|         if outside:
  190|             raise ValueError(f"keep_index has entries outside [0, {width}): {outside}")
  191|         if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
  192|             raise ValueError(
  193|                 "keep_index must be strictly ascending; the delay vector is positional against "
  194|                 "it, so a reordered index would delay the wrong channels."
  195|             )
  196| 
  197|         self.declared_width = width
  202|         self.register_buffer(
  203|             "keep_index", torch.tensor(indices, dtype=torch.long), persistent=False
  204|         )
  205|         self.delay = ChannelDelay(
  206|             num_channels=len(indices),
  207|             delays=[0] * len(indices) if delays is None else delays,
  208|         )
  209| 
  210|     @property
  211|     def out_channels(self) -> int:
  213|         return int(self.keep_index.numel())
  214| 
  215|     @property
  216|     def max_delay(self) -> int:
  218|         return self.delay.max_delay
  219| 
  220|     def extra_repr(self) -> str:
  222|         return f"{self.declared_width} -> {self.out_channels}, max_delay={self.max_delay}"
  223| 
  224|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  233|         return self.delay(torch.index_select(x, -1, self.keep_index))
```

## 4. Input adapters, conv-LSTM encoder, conv stem — `teb_vae/lag_attn/nets/encoders.py`

`AvailabilityInputAdapter`: `Linear(in→128)` on masked input + `mask_proj(availability−1)` + `start_embed` where no channel is available, then LayerNorm → GELU → Dropout → `ResidualMLP(128→(128,128,128,128), final_activation=False)`. `CausalConvLstmEncoder`: `front_mlp` → [conv stack ‖ LSTM] → `fusion ResidualMLP(256→…→128)` → LayerNorm. `CausalConvStem`: conv stack + LayerNorm(out + x), receptive field $1+\sum(k-1)r$ = 387 steps at the shipped schedule.

`teb_vae/lag_attn/nets/encoders.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   14| from __future__ import annotations
   15| 
   16| from typing import List, Optional, Sequence, Tuple
   17| 
   18| import torch
   19| from torch import nn
   20| 
   21| from teb_vae.lag_attn.nets.blocks import (
   22|     CausalMultiChannelConvBlock,
   23|     ResidualMLP,
   24|     geometric_schedule,
   25| )
   26| 
   30| START_EMBED_STD = 0.02
   31| 
   33| class InputAdapter(nn.Module):
   44| 
   45|     def __init__(
   46|         self,
   47|         in_dim: int,
   48|         d_model: int = 128,
   49|         dropout: float = 0.1,
   50|         post_residual_activation: bool = True,
   51|     ) -> None:
   64|         super().__init__()
   65|         self.linear = nn.Linear(in_dim, d_model)
   66|         self.norm = nn.LayerNorm(d_model)
   67|         self.act = nn.GELU()
   68|         self.drop = nn.Dropout(dropout)
   69|         self.res_mlp = ResidualMLP(
   70|             input_dim=d_model,
   71|             hidden_dims=geometric_schedule(d_model, d_model, 3),
   72|             final_activation=post_residual_activation,
   73|             use_skip_connection=True,
   74|             use_input_layer_norm=True,
   75|             activation=nn.GELU,
   76|             dropout=dropout,
   77|         )
   78| 
   79|     def forward(self, x: torch.Tensor) -> torch.Tensor:
   88|         x = self.linear(x)
   89|         x = self.norm(x)
   90|         x = self.act(x)
   91|         x = self.drop(x)
   92|         return self.res_mlp(x)
   93| 
   95| class AvailabilityInputAdapter(nn.Module):
  161| 
  166|     availability: torch.Tensor
  167|     start_indicator: torch.Tensor
  168| 
  169|     def __init__(
  170|         self,
  171|         *,
  172|         in_dim: int,
  173|         d_model: int,
  174|         sequence_length: int,
  175|         dropout: float = 0.1,
  176|         delays: Optional[Sequence[int]] = None,
  177|         post_residual_activation: bool = False,
  178|     ) -> None:
  203|         super().__init__()
  204|         self.in_dim = int(in_dim)
  205|         self.d_model = int(d_model)
  206|         self.sequence_length = int(sequence_length)
  207| 
  209|         self.linear = nn.Linear(in_dim, d_model)
  210|         self.norm = nn.LayerNorm(d_model)
  211|         self.act = nn.GELU()
  212|         self.drop = nn.Dropout(dropout)
  213|         self.res_mlp = ResidualMLP(
  214|             input_dim=d_model,
  215|             hidden_dims=geometric_schedule(d_model, d_model, 3),
  216|             final_activation=post_residual_activation,
  217|             use_skip_connection=True,
  218|             use_input_layer_norm=True,
  219|             activation=nn.GELU,
  220|             dropout=dropout,
  221|         )
  222| 
  223|         delay_values = self._validate_delays(delays, self.in_dim)
  224|         self.max_delay = max(delay_values) if delay_values else 0
  225|         self.min_delay = min(delay_values) if delay_values else 0
  226| 
  227|         self.mask_proj: Optional[nn.Linear] = None
  228|         self.start_embed: Optional[nn.Parameter] = None
  229|         if self.max_delay > 0:
  230|             pattern = self._availability_pattern(delay_values, self.sequence_length)
  235|             self.register_buffer("availability", pattern, persistent=False)
  236|             self.mask_proj = nn.Linear(self.in_dim, d_model, bias=False)
  237|             if self.min_delay > 0:
  238|                 indicator = (pattern.sum(dim=-1) == 0).to(pattern.dtype).unsqueeze(-1)
  239|                 self.register_buffer("start_indicator", indicator, persistent=False)
  240|                 self.start_embed = nn.Parameter(torch.randn(d_model) * START_EMBED_STD)
  241| 
  242|     @staticmethod
  243|     def _validate_delays(delays: Optional[Sequence[int]], in_dim: int) -> List[int]:
  256|         if delays is None:
  257|             return []
  258|         values = [int(value) for value in delays]
  259|         if len(values) != in_dim:
  260|             raise ValueError(
  261|                 f"delays has {len(values)} entries but the adapter reads {in_dim} channels; the "
  262|                 f"delay vector is positional against the surviving channels, so a length mismatch "
  263|                 f"would mark the wrong channels unavailable with no other failure signal"
  264|             )
  265|         negative = [(index, value) for index, value in enumerate(values) if value < 0]
  266|         if negative:
  267|             raise ValueError(
  268|                 f"delays must be >= 0; got negative entries at {negative}"
  269|             )
  270|         return values
  271| 
  272|     @staticmethod
  273|     def _availability_pattern(delays: Sequence[int], sequence_length: int) -> torch.Tensor:
  284|         steps = torch.arange(sequence_length).unsqueeze(-1)
  285|         return (steps >= torch.tensor(list(delays), dtype=torch.long)).to(torch.float32)
  286| 
  287|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  302|         seq_len = int(x.shape[1])
  303|         self._validate_stream(x)
  304| 
  309|         if self.mask_proj is not None:
  310|             available = self._slice(self.availability, seq_len)
  311|             embedded = self.linear(x * available) + self.mask_proj(available - 1.0)
  312|         else:
  313|             embedded = self.linear(x)
  314|         if self.start_embed is not None:
  315|             embedded = embedded + self._slice(self.start_indicator, seq_len) * self.start_embed
  316| 
  317|         return self.res_mlp(self.drop(self.act(self.norm(embedded))))
  318| 
  319|     def _validate_stream(self, x: torch.Tensor) -> None:
  340|         if x.dim() != 3 or int(x.shape[-1]) != self.in_dim:
  341|             raise ValueError(
  342|                 f"the stream is {tuple(x.shape)} but this adapter reads {self.in_dim} channels; "
  343|                 f"the availability pattern is positional against that width and broadcasts over a "
  344|                 f"narrower one rather than refusing it"
  345|             )
  346| 
  347|     def _slice(self, pattern: torch.Tensor, seq_len: int) -> torch.Tensor:
  360|         if seq_len > self.sequence_length:
  361|             raise ValueError(
  362|                 f"sequence of {seq_len} steps exceeds the availability pattern built for "
  363|                 f"sequence_length={self.sequence_length}"
  364|             )
  365|         return pattern[:seq_len]
  366| 
  367|     def extra_repr(self) -> str:
  369|         terms = [
  370|             name
  371|             for name, built in (
  372|                 ("W_m", self.mask_proj is not None),
  373|                 ("e_start", self.start_embed is not None),
  374|             )
  375|             if built
  376|         ]
  377|         return (
  378|             f"{self.in_dim} -> {self.d_model}, max_delay={self.max_delay}, "
  379|             f"availability_terms={terms or 'none'}"
  380|         )
  381| 
  383| class CausalConvLstmEncoder(nn.Module):
  404| 
  405|     def __init__(
  406|         self,
  407|         *,
  408|         d_model: int,
  409|         cnn_kernels: Tuple[int, ...],
  410|         cnn_dilations: Tuple[int, ...],
  411|         lstm_layers: int,
  412|         lstm_dropout: float,
  413|         conv_dropout: float,
  414|         stack_skip_connection: bool = True,
  415|         post_residual_activation: bool = True,
  416|         conv_norm_groups: Optional[int] = None,
  417|     ) -> None:
  447|         super().__init__()
  448|         self.d_model = d_model
  449| 
  450|         if len(cnn_kernels) != len(cnn_dilations):
  451|             raise ValueError(
  452|                 "cnn_kernels and cnn_dilations must have equal length, got "
  453|                 f"{len(cnn_kernels)} and {len(cnn_dilations)}"
  454|             )
  455|         if len(cnn_kernels) < 1:
  456|             raise ValueError("need at least one causal conv block")
  457| 
  458|         self.post_residual_activation = bool(post_residual_activation)
  459| 
  461|         self.front_mlp = ResidualMLP(
  462|             input_dim=d_model,
  463|             hidden_dims=geometric_schedule(d_model, d_model, 3),
  464|             final_activation=self.post_residual_activation,
  465|             use_skip_connection=True,
  466|             use_input_layer_norm=True,
  467|             activation=nn.GELU,
  468|             dropout=conv_dropout,
  469|         )
  470| 
  473|         self.convs = nn.ModuleList(
  474|             [
  475|                 CausalMultiChannelConvBlock(
  476|                     in_channels=d_model,
  477|                     out_channels=d_model,
  478|                     filter_size=kernel,
  479|                     dilation=dilation,
  480|                     dropout=conv_dropout,
  481|                     activation=nn.GELU,
  482|                     norm_groups=conv_norm_groups,
  483|                 )
  484|                 for kernel, dilation in zip(cnn_kernels, cnn_dilations)
  485|             ]
  486|         )
  487|         self.stack_skip_connection = bool(stack_skip_connection)
  492|         self.stack_skip_norms: Optional[nn.ModuleList]
  493|         if self.stack_skip_connection:
  494|             self.stack_skip_norms = nn.ModuleList(
  495|                 [
  496|                     nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
  497|                     for _ in range(len(self.convs) - 1)
  498|                 ]
  499|             )
  500|         else:
  501|             self.stack_skip_norms = None
  502|         self.conv_out_norm = nn.LayerNorm(d_model)
  503| 
  505|         self.lstm = nn.LSTM(
  506|             input_size=d_model,
  507|             hidden_size=d_model,
  508|             num_layers=lstm_layers,
  509|             batch_first=True,
  510|             bidirectional=False,
  511|             dropout=lstm_dropout if lstm_layers > 1 else 0.0,
  512|         )
  513|         self.lstm_norm = nn.LayerNorm(d_model)
  514| 
  516|         self.fusion = ResidualMLP(
  517|             input_dim=2 * d_model,
  518|             hidden_dims=geometric_schedule(2 * d_model, d_model, 3),
  519|             final_activation=self.post_residual_activation,
  520|             use_skip_connection=True,
  521|             use_input_layer_norm=True,
  522|             activation=nn.GELU,
  523|             dropout=conv_dropout,
  524|         )
  528|         self.output_norm = nn.LayerNorm(d_model)
  529| 
  530|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  539|         x_lin = self.front_mlp(x)
  540| 
  542|         x_conv = x_lin.transpose(1, 2).contiguous()
  543|         out = self.convs[0](x_conv)
  544|         skip_norms = self.stack_skip_norms
  545|         for index in range(1, len(self.convs)):
  546|             block_out = self.convs[index](out)
  547|             if skip_norms is not None:
  552|                 block_out = block_out + skip_norms[index - 1](out)
  553|             out = block_out
  554|         conv_out = self.conv_out_norm(out.transpose(1, 2).contiguous() + x_lin)
  555| 
  556|         lstm_out, _ = self.lstm(x_lin)
  557|         lstm_out = self.lstm_norm(lstm_out)
  558| 
  559|         fused = torch.cat([conv_out, lstm_out], dim=-1)
  560|         return self.output_norm(self.fusion(fused))
  561| 
  581| LAG_KV_SOURCE_CHOICES: Tuple[str, ...] = ("encoder", "conv_stem", "adapter")
  582| 
  584| class CausalConvStem(nn.Module):
  605| 
  606|     def __init__(
  607|         self,
  608|         *,
  609|         d_model: int,
  610|         cnn_kernels: Tuple[int, ...],
  611|         cnn_dilations: Tuple[int, ...],
  612|         conv_dropout: float,
  613|         conv_norm_groups: Optional[int] = None,
  614|     ) -> None:
  631|         super().__init__()
  632|         if len(cnn_kernels) != len(cnn_dilations):
  633|             raise ValueError(
  634|                 "cnn_kernels and cnn_dilations must have equal length, got "
  635|                 f"{len(cnn_kernels)} and {len(cnn_dilations)}"
  636|             )
  637|         if len(cnn_kernels) < 1:
  638|             raise ValueError(
  639|                 "need at least one causal conv block; a stem of zero blocks is the identity, and "
  640|                 "the identity representation is the 'adapter' arm, which is chosen by name"
  641|             )
  642| 
  643|         self.d_model = int(d_model)
  644|         self.cnn_kernels = tuple(int(kernel) for kernel in cnn_kernels)
  645|         self.cnn_dilations = tuple(int(dilation) for dilation in cnn_dilations)
  646|         self.convs = nn.ModuleList(
  647|             [
  648|                 CausalMultiChannelConvBlock(
  649|                     in_channels=d_model,
  650|                     out_channels=d_model,
  651|                     filter_size=kernel,
  652|                     dilation=dilation,
  653|                     dropout=conv_dropout,
  654|                     activation=nn.GELU,
  655|                     norm_groups=conv_norm_groups,
  656|                 )
  657|                 for kernel, dilation in zip(self.cnn_kernels, self.cnn_dilations)
  658|             ]
  659|         )
  664|         self.output_norm = nn.LayerNorm(d_model)
  665| 
  666|     @property
  667|     def receptive_field(self) -> int:
  676|         return 1 + sum(
  677|             (kernel - 1) * dilation
  678|             for kernel, dilation in zip(self.cnn_kernels, self.cnn_dilations)
  679|         )
  680| 
  681|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  691|         out = x.transpose(1, 2).contiguous()
  692|         for block in self.convs:
  693|             out = block(out)
  694|         return self.output_norm(out.transpose(1, 2).contiguous() + x)
  695| 
  696|     def extra_repr(self) -> str:
  698|         return (
  699|             f"d_model={self.d_model}, kernels={self.cnn_kernels}, "
  700|             f"dilations={self.cnn_dilations}, receptive_field={self.receptive_field} steps"
  701|         )
```

## 5. Lag cross-attention — `teb_vae/lag_attn/nets/attention.py`

Keys/values are unfolded into a $(B,T,M,d,L)$ window of the previous $L=91$ steps; window index $j$ ↔ lag $L-1-j$ (hence the `.flip` calls). Scores = content + Shaw relative key bias (`lag_embeddings`), scaled by $1/\sqrt{d}$, plus the unscaled `lag_score_bias` (ALiBi-shaped, learnable; all zeros at `alibi_slope_scale=0`). `entmax15` or softmax over lags; masked lags $-\infty$; all-masked rows → 0. Returns `W_o(head_out)` (frozen, unused), `alpha (B,T,M,L)` in lag order, `head_out (B,T,M,d_head)`.

`teb_vae/lag_attn/nets/attention.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   23| from __future__ import annotations
   24| 
   25| import math
   26| from typing import Optional, Tuple, cast
   27| 
   28| import torch
   29| import torch.nn.functional as F
   30| from entmax import entmax15
   31| from torch import nn
   32| from torch.utils.checkpoint import checkpoint
   33| 
   35| def alibi_slopes(num_heads: int) -> torch.Tensor:
   50| 
   51|     def _pow2_slopes(n: int) -> list:
   52|         start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
   53|         return [start * (start**i) for i in range(n)]
   54| 
   55|     if math.log2(num_heads).is_integer():
   56|         slopes = _pow2_slopes(num_heads)
   57|     else:  # pragma: no cover - num_heads is a power of two in practice
   58|         closest = 2 ** int(math.floor(math.log2(num_heads)))
   59|         slopes = _pow2_slopes(closest)
   60|         extra = _pow2_slopes(2 * closest)[0::2][: num_heads - closest]
   61|         slopes = slopes + extra
   62|     return torch.tensor(slopes, dtype=torch.float32)
   63| 
   65| class LagCrossAttention(nn.Module):
   81| 
   82|     def __init__(
   83|         self,
   84|         d_model: int = 128,
   85|         num_heads: int = 4,
   86|         d_head: int = 32,
   87|         max_lag: int = 90,
   88|         dropout: float = 0.1,
   89|         use_entmax: bool = False,
   90|         grad_checkpoint: bool = False,
   91|         lag_bias_init: str = "normal",
   92|         alibi_slope_scale: float = 1.0,
   93|     ) -> None:
  117|         super().__init__()
  118|         if num_heads * d_head != d_model:
  119|             raise ValueError(
  120|                 f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
  121|             )
  122|         if lag_bias_init not in ("normal", "alibi_decay"):
  123|             raise ValueError(
  124|                 f"lag_bias_init must be 'normal' or 'alibi_decay', got {lag_bias_init!r}"
  125|             )
  126| 
  127|         self.d_model = d_model
  128|         self.num_heads = num_heads
  129|         self.d_head = d_head
  130|         self.L = int(max_lag) + 1
  131|         self.scale = 1.0 / math.sqrt(d_head)
  132|         self.use_entmax = bool(use_entmax)
  133|         self.grad_checkpoint = bool(grad_checkpoint)
  134| 
  137|         self.q_norm = nn.LayerNorm(d_model)
  138|         self.kv_norm = nn.LayerNorm(d_model)
  139| 
  140|         self.W_q = nn.Linear(d_model, d_model)
  141|         self.W_k = nn.Linear(d_model, d_model)
  142|         self.W_v = nn.Linear(d_model, d_model)
  143|         self.W_o = nn.Linear(d_model, d_model)
  144|         self.attn_dropout = nn.Dropout(dropout)
  145| 
  147|         self.lag_embeddings = nn.Parameter(torch.zeros(self.L, num_heads, d_head))
  148|         nn.init.normal_(self.lag_embeddings, mean=0.0, std=0.02)
  149| 
  150|         self.lag_bias_init = lag_bias_init
  151|         self.alibi_slope_scale = float(alibi_slope_scale)
  152|         if lag_bias_init == "alibi_decay":
  153|             slopes = alibi_slopes(num_heads) * float(alibi_slope_scale)  # (num_heads,)
  154|             lags = torch.arange(self.L, dtype=torch.float32)             # (L,)
  155|             decay = -slopes[:, None] * lags[None, :]                     # (num_heads, L)
  156|             self.lag_score_bias = nn.Parameter(decay)
  157|         else:
  158|             self.register_parameter("lag_score_bias", None)
  159| 
  160|     def build_lag_mask(
  161|         self, seq_len: int, device: Optional[torch.device] = None
  162|     ) -> torch.Tensor:
  179|         steps = torch.arange(seq_len, device=device)[:, None]
  180|         lags = torch.arange(self.L, device=device)[None, :]
  181|         return steps - lags >= 0
  182| 
  183|     def _attend(
  184|         self,
  185|         h_y: torch.Tensor,
  186|         h_u: torch.Tensor,
  187|         m_lag: torch.Tensor,
  188|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  201|         batch, seq_len = h_y.shape[0], h_y.shape[1]
  202|         num_lags = self.L
  203|         heads = self.num_heads
  204|         width = self.d_head
  205| 
  207|         q = self.W_q(self.q_norm(h_y)).view(batch, seq_len, heads, width)
  208|         h_u_normed = self.kv_norm(h_u)
  209|         k = self.W_k(h_u_normed).view(batch, seq_len, heads, width)
  210|         v = self.W_v(h_u_normed).view(batch, seq_len, heads, width)
  211| 
  215|         k_padded = F.pad(k, (0, 0, 0, 0, num_lags - 1, 0))  # (B, T+L-1, Mh, d)
  216|         v_padded = F.pad(v, (0, 0, 0, 0, num_lags - 1, 0))
  217|         k_window = k_padded.unfold(1, num_lags, 1)          # (B, T, Mh, d, L) view
  218|         v_window = v_padded.unfold(1, num_lags, 1)          # (B, T, Mh, d, L) view
  219| 
  221|         scores = torch.einsum("btmd,btmdj->btmj", q, k_window)
  222|         scores = scores + torch.einsum("btmd,jmd->btmj", q, self.lag_embeddings.flip(0))
  223|         scores = scores * self.scale
  224|         if self.lag_score_bias is not None:
  225|             scores = scores + self.lag_score_bias.flip(-1)[None, None, :, :]
  226| 
  228|         mask_window = m_lag.flip(-1).to(torch.bool)
  229|         scores = scores.masked_fill(~mask_window[None, :, None, :], float("-inf"))
  230| 
  232|         alpha_window = (
  233|             cast(torch.Tensor, entmax15(scores, dim=-1))
  234|             if self.use_entmax
  235|             else F.softmax(scores, dim=-1)
  236|         )
  240|         alpha_window = torch.nan_to_num(alpha_window, nan=0.0)
  241|         alpha_window = self.attn_dropout(alpha_window)
  242| 
  243|         head_out = torch.einsum("btmj,btmdj->btmd", alpha_window, v_window)
  244|         out = self.W_o(head_out.reshape(batch, seq_len, heads * width))
  245| 
  246|         alpha = alpha_window.flip(-1)  # back to lag order
  249|         return out, alpha, head_out
  250| 
  251|     def forward(
  252|         self,
  253|         h_y: torch.Tensor,
  254|         h_u: torch.Tensor,
  255|         m_lag: Optional[torch.Tensor] = None,
  256|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  271|         if m_lag is None:
  272|             m_lag = self.build_lag_mask(h_y.shape[1], device=h_y.device)
  273|         elif m_lag.dim() == 3:
  274|             m_lag = m_lag[0]
  275|         if self.grad_checkpoint and self.training:
  276|             return cast(
  277|                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  278|                 checkpoint(self._attend, h_y, h_u, m_lag, use_reentrant=False),
  279|             )
  280|         return self._attend(h_y, h_u, m_lag)
```

## 6. Latent heads

### 6.1 `teb_vae/lag_attn/nets/heads.py` — `PriorHead` (ancestor only, unused by these four), `PosteriorHead`, `TEAnalysisHead`

Head-structured posterior (`head_structured=True` in every model here): for head $m$, `fusion[m](cat[LN(h_y), a_dropout(LN_gain(a[:,:,m,:]))]) → (B,T,32)`, `delta_mu_head[m] → (B,T,16)`; concatenated over heads → `(B,T,64)`; $\mu^q=\mu^p+3\tanh(\cdot/3)$. `posterior_logvar_mode='independent'` (shipped): `logvar_post_head[m]` on the same fused features → `smooth_bound`. `TEAnalysisHead`: `kld_per_head = kld_btd.view(B,T,M,16).sum(-1)`, `te_lag_map = einsum(kld_per_head, alpha)`.

`teb_vae/lag_attn/nets/heads.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   18| from __future__ import annotations
   19| 
   20| from typing import List, Optional, Tuple, cast
   21| 
   22| import torch
   23| from torch import nn
   24| 
   25| from teb_vae.lag_attn.nets.blocks import (
   26|     ResidualMLP,
   27|     geometric_schedule,
   28|     smooth_bound,
   29|     validate_choice,
   30| )
   31| 
   36| POSTERIOR_LOGVAR_MODES = ("residual", "independent")
   37| 
   39| class PriorHead(nn.Module):
   57| 
   58|     def __init__(
   59|         self,
   60|         d_model: int = 128,
   61|         d_z: int = 24,
   62|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
   63|         dropout: float = 0.1,
   64|         mu_scale: float = 5.0,
   65|     ) -> None:
   79|         super().__init__()
   80|         if mu_scale <= 0.0:
   81|             raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
   82|         self.logvar_clamp = logvar_clamp
   83|         self.mu_scale = float(mu_scale)
   84| 
   86|         self.mu_input_norm = nn.LayerNorm(d_model)
   87|         self.logvar_input_norm = nn.LayerNorm(d_model)
   88|         self.dec_input_norm = nn.LayerNorm(d_model)
   89| 
   90|         self.mu_prior_head = ResidualMLP(
   91|             input_dim=d_model,
   92|             hidden_dims=geometric_schedule(d_model, d_z, 4),
   93|             final_activation=False,
   94|             use_skip_connection=True,
   95|             use_input_layer_norm=True,
   96|             activation=nn.GELU,
   97|             dropout=dropout,
   98|         )
   99|         self.logvar_prior_head = ResidualMLP(
  100|             input_dim=d_model,
  101|             hidden_dims=geometric_schedule(d_model, d_z, 4),
  102|             final_activation=False,
  103|             use_skip_connection=True,
  104|             use_input_layer_norm=True,
  105|             activation=nn.GELU,
  106|             dropout=dropout,
  107|         )
  108|         self.decoder_state_head = ResidualMLP(
  109|             input_dim=d_model,
  110|             hidden_dims=geometric_schedule(d_model, d_model, 3),
  111|             final_activation=True,
  112|             use_skip_connection=True,
  113|             use_input_layer_norm=True,
  114|             activation=nn.GELU,
  115|             dropout=dropout,
  116|         )
  117| 
  118|     def forward(
  119|         self, h_y: torch.Tensor
  120|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  129|         raw_mu = self.mu_prior_head(self.mu_input_norm(h_y))
  130|         mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)
  131| 
  132|         raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(h_y))
  133|         logvar_prior = smooth_bound(raw_logvar_prior, *self.logvar_clamp)
  134| 
  135|         decoder_state = self.decoder_state_head(self.dec_input_norm(h_y))
  136|         return mu_prior, logvar_prior, decoder_state, raw_logvar_prior
  137| 
  139| class PosteriorHead(nn.Module):
  160| 
  161|     def __init__(
  162|         self,
  163|         d_model: int = 128,
  164|         d_z: int = 24,
  165|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  166|         dropout: float = 0.1,
  167|         delta_mu_scale: float = 3.0,
  168|         head_structured: bool = False,
  169|         num_heads: int = 4,
  170|         d_head: int = 32,
  171|         *,
  172|         delta_logvar_scale: float = 2.0,
  173|         posterior_logvar_mode: str = "residual",
  174|         source_dropout: float = 0.0,
  175|     ) -> None:
  224|         super().__init__()
  225|         if delta_mu_scale <= 0.0:
  226|             raise ValueError(f"delta_mu_scale must be > 0, got {delta_mu_scale}")
  227|         if delta_logvar_scale <= 0.0:
  228|             raise ValueError(f"delta_logvar_scale must be > 0, got {delta_logvar_scale}")
  229| 
  230|         self.logvar_clamp = logvar_clamp
  231|         self.delta_mu_scale = float(delta_mu_scale)
  232|         self.delta_logvar_scale = float(delta_logvar_scale)
  233|         self.head_structured = bool(head_structured)
  234|         self.num_heads = int(num_heads)
  235|         self.posterior_logvar_mode = validate_choice(
  236|             posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
  237|         )
  238| 
  240|         self.h_y_norm = nn.LayerNorm(d_model)
  250|         self.a_dropout = nn.Dropout(float(source_dropout))
  251| 
  252|         if self.head_structured:
  253|             if d_z % num_heads != 0:
  254|                 raise ValueError(
  255|                     f"head_structured posterior needs d_z % num_heads == 0, "
  256|                     f"got d_z={d_z}, num_heads={num_heads}"
  257|                 )
  258|             self.group = d_z // num_heads
  259|             self.d_head = int(d_head)
  260|             self.a_head_norm = nn.LayerNorm(d_head)
  261|             fuse_in = d_model + d_head
  262|             fuse_out = max(2 * self.group, 16)
  264|             self.fusion = nn.ModuleList(
  265|                 [
  266|                     ResidualMLP(
  267|                         input_dim=fuse_in,
  268|                         hidden_dims=geometric_schedule(fuse_in, fuse_out, 2),
  269|                         final_activation=True,
  270|                         use_skip_connection=True,
  271|                         use_input_layer_norm=True,
  272|                         activation=nn.GELU,
  273|                         dropout=dropout,
  274|                     )
  275|                     for _ in range(num_heads)
  276|                 ]
  277|             )
  278|             self.delta_mu_head = nn.ModuleList(
  279|                 [nn.Linear(fuse_out, self.group) for _ in range(num_heads)]
  280|             )
  285|             logvar_head: nn.Module = nn.ModuleList(
  286|                 [nn.Linear(fuse_out, self.group) for _ in range(num_heads)]
  287|             )
  288|         else:
  290|             self.a_norm = nn.LayerNorm(d_model)
  291|             fused_in = 2 * d_model
  292|             self.fusion = ResidualMLP(
  293|                 input_dim=fused_in,
  294|                 hidden_dims=geometric_schedule(fused_in, d_model, 3),
  295|                 final_activation=True,
  296|                 use_skip_connection=True,
  297|                 use_input_layer_norm=True,
  298|                 activation=nn.GELU,
  299|                 dropout=dropout,
  300|             )
  301|             self.delta_mu_head = nn.Linear(d_model, d_z)
  302|             logvar_head = nn.Linear(d_model, d_z)
  303| 
  307|         self.delta_logvar_head = (
  308|             logvar_head if self.posterior_logvar_mode == "residual" else None
  309|         )
  310|         self.logvar_post_head = (
  311|             logvar_head if self.posterior_logvar_mode == "independent" else None
  312|         )
  313| 
  314|     def forward(
  315|         self,
  316|         h_y: torch.Tensor,
  317|         a: torch.Tensor,
  318|         mu_prior: torch.Tensor,
  319|         raw_logvar_prior: Optional[torch.Tensor] = None,
  320|     ) -> Tuple[torch.Tensor, torch.Tensor]:
  340|         if raw_logvar_prior is None:
  341|             raise ValueError(
  342|                 "the posterior log-variance is built against the prior's pre-bound raw "
  343|                 "log-variance, so raw_logvar_prior is required; call via the model's forward "
  344|                 "or encode_only, which thread it through"
  345|             )
  346| 
  347|         fused_heads: List[torch.Tensor] = []
  348|         fused_flat: Optional[torch.Tensor] = None
  349|         if self.head_structured:
  351|             fusion = cast(nn.ModuleList, self.fusion)
  352|             delta_mu_head = cast(nn.ModuleList, self.delta_mu_head)
  353| 
  354|             h_y_normed = self.h_y_norm(h_y)
  355|             raw_deltas = []
  356|             for index in range(self.num_heads):
  357|                 a_head = self.a_dropout(self.a_head_norm(a[:, :, index, :]))
  358|                 fused_head = fusion[index](torch.cat([h_y_normed, a_head], dim=-1))
  359|                 fused_heads.append(fused_head)
  360|                 raw_deltas.append(delta_mu_head[index](fused_head))
  361|             raw_delta = torch.cat(raw_deltas, dim=-1)
  362|         else:
  363|             fused_flat = self.fusion(
  364|                 torch.cat([self.h_y_norm(h_y), self.a_dropout(self.a_norm(a))], dim=-1)
  365|             )
  366|             raw_delta = self.delta_mu_head(fused_flat)
  367| 
  370|         delta_mu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
  371|         mu_post = mu_prior + delta_mu
  372| 
  373|         raw_logvar_post = self._run_logvar_head(fused_heads, fused_flat)
  374|         if self.delta_logvar_head is not None:
  375|             delta_logvar = (
  376|                 self.delta_logvar_scale * torch.tanh(raw_logvar_post / self.delta_logvar_scale)
  377|             )
  380|             logvar_post = smooth_bound(raw_logvar_prior + delta_logvar, *self.logvar_clamp)
  381|         else:
  385|             logvar_post = smooth_bound(raw_logvar_post, *self.logvar_clamp)
  386| 
  387|         return mu_post, logvar_post
  388| 
  389|     def _run_logvar_head(
  390|         self, fused_heads: List[torch.Tensor], fused_flat: Optional[torch.Tensor]
  391|     ) -> torch.Tensor:
  406|         head = self.delta_logvar_head if self.logvar_post_head is None else self.logvar_post_head
  407|         if self.head_structured:
  408|             modules = cast(nn.ModuleList, head)
  409|             return torch.cat(
  410|                 [modules[index](fused_heads[index]) for index in range(self.num_heads)], dim=-1
  411|             )
  412|         return cast(nn.Linear, head)(fused_flat)
  413| 
  415| class TEAnalysisHead(nn.Module):
  429| 
  430|     def forward(
  431|         self,
  432|         kld_btd: torch.Tensor,
  433|         attn_weights: torch.Tensor,
  434|         head_structured: bool = False,
  435|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  448|         batch, seq_len, d_z = kld_btd.shape
  449|         num_heads = attn_weights.shape[2]
  450|         kld_per_t = kld_btd.sum(dim=-1)
  451| 
  452|         if d_z % num_heads == 0:
  453|             group = d_z // num_heads
  454|             kld_per_head = kld_btd.view(batch, seq_len, num_heads, group).sum(dim=-1)
  455|         else:  # pragma: no cover - d_z is divisible by num_heads in practice
  456|             kld_per_head = kld_per_t.unsqueeze(-1).expand(batch, seq_len, num_heads) / num_heads
  457| 
  458|         if head_structured and d_z % num_heads == 0:
  459|             te_lag_map = torch.einsum("btm,btml->btl", kld_per_head, attn_weights)
  460|         else:
  461|             mean_alpha = attn_weights.mean(dim=-2)
  462|             te_lag_map = kld_per_t.unsqueeze(-1) * mean_alpha
  463| 
  464|         return kld_per_t, te_lag_map, kld_per_head
```

### 6.2 `teb_vae/lag_attn_rws/nets/heads.py` — `FullLatentPriorHead`

Two `ResidualMLP(128→(111,97,84,74,64), no final act)` on LayerNorm'd $h_y$ (+ `clock_proj(LN(clock))` added to $h_y$ when built); $\mu^p=5\tanh(\cdot/5)$, $\ell^p=\mathrm{smooth\_bound}(\cdot,-5,3)$; also returns the raw pre-bound log-variance.

`teb_vae/lag_attn_rws/nets/heads.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   38| from __future__ import annotations
   39| 
   40| from typing import Optional, Tuple
   41| 
   42| import torch
   43| from torch import nn
   44| 
   45| from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound
   46| 
   48| class FullLatentPriorHead(nn.Module):
   62| 
   67|     clock_norm: Optional[nn.LayerNorm]
   68|     clock_proj: Optional[nn.Linear]
   69| 
   70|     def __init__(
   71|         self,
   72|         d_model: int = 128,
   73|         d_z: int = 48,
   74|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
   75|         dropout: float = 0.1,
   76|         mu_scale: float = 5.0,
   77|         clock_dim: Optional[int] = None,
   78|     ) -> None:
   98|         super().__init__()
   99|         if mu_scale <= 0.0:
  100|             raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
  101|         self.logvar_clamp = logvar_clamp
  102|         self.mu_scale = float(mu_scale)
  103| 
  105|         self.mu_input_norm = nn.LayerNorm(d_model)
  106|         self.logvar_input_norm = nn.LayerNorm(d_model)
  107| 
  118|         self.clock_norm = None
  119|         self.clock_proj = None
  120|         if clock_dim is not None:
  121|             if int(clock_dim) <= 0:
  122|                 raise ValueError(
  123|                     f"clock_dim must be > 0 when given, got {clock_dim}. Pass None for a head "
  124|                     f"that conditions on the target state alone; a zero-width clock builds a "
  125|                     f"projection no forward can reach."
  126|                 )
  127|             self.clock_norm = nn.LayerNorm(int(clock_dim))
  128|             self.clock_proj = nn.Linear(int(clock_dim), d_model, bias=False)
  129|             self.zero_init_clock()
  130| 
  131|         self.mu_prior_head = ResidualMLP(
  132|             input_dim=d_model,
  133|             hidden_dims=geometric_schedule(d_model, d_z, 4),
  134|             final_activation=False,
  135|             use_skip_connection=True,
  136|             use_input_layer_norm=True,
  137|             activation=nn.GELU,
  138|             dropout=dropout,
  139|         )
  140|         self.logvar_prior_head = ResidualMLP(
  141|             input_dim=d_model,
  142|             hidden_dims=geometric_schedule(d_model, d_z, 4),
  143|             final_activation=False,
  144|             use_skip_connection=True,
  145|             use_input_layer_norm=True,
  146|             activation=nn.GELU,
  147|             dropout=dropout,
  148|         )
  149| 
  150|     def zero_init_clock(self) -> None:
  161|         if self.clock_proj is not None:
  162|             nn.init.zeros_(self.clock_proj.weight)
  163| 
  164|     def forward(
  165|         self, h_y: torch.Tensor, clock: Optional[torch.Tensor] = None
  166|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  190|         if (self.clock_proj is None) != (clock is None):
  191|             raise ValueError(
  192|                 f"the prior head was built {'with' if self.clock_proj is not None else 'without'} "
  193|                 f"a clock path and was called {'with' if clock is not None else 'without'} a "
  194|                 f"clock. The two are one decision: prior_availability_input builds the projection "
  195|                 f"and the forward that supplies it, and half of that is a model whose KL means "
  196|                 f"something other than what its configuration says."
  197|             )
  201|         features = h_y
  202|         if self.clock_proj is not None and self.clock_norm is not None:
  203|             features = h_y + self.clock_proj(self.clock_norm(clock))
  204| 
  205|         raw_mu = self.mu_prior_head(self.mu_input_norm(features))
  206|         mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)
  207| 
  208|         raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(features))
  209|         logvar_prior = smooth_bound(raw_logvar_prior, *self.logvar_clamp)
  210|         return mu_prior, logvar_prior, raw_logvar_prior
```

## 7. Horizon decoder — `teb_vae/lag_attn/nets/decoders.py`

`BaselineFutureDecoder.forward(z (B,A,64), persistence (B,A,C)|None)`: `proj ResidualMLP(64→(91,128,181,256), GELU)`; `core.decode`: broadcast over $H$ + `horizon_embedding (30,256)`; fold to `(B·A, 256, 30)`; `_HorizonRefine` 4 blocks `Conv1d(256,256,k=3, dil 1,2,4,8, symmetric pad)` → GroupNorm(8) → GELU → per-block FiLM $(1+\gamma)y+\beta$ from `film[i](z_proj)` → residual; 2 × `_HorizonSelfAttention` over the 30 horizon tokens (pre-LN, bias-free q/k/v/out, SDPA, residual gain init 0.01); `out_norm(feat + skip)`. Heads: `mean_head Linear(256→C)` (+ `persistence_weight (30,C) * y_anchor`), `logvar_head → smooth_bound(-5,3)`. `ResidualFutureDecoder` is the ancestor's second decoder; none of the four models build it.

`teb_vae/lag_attn/nets/decoders.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   31| from __future__ import annotations
   32| 
   33| from typing import Optional, Tuple, cast
   34| 
   35| import torch
   36| import torch.nn.functional as F
   37| from torch import nn
   38| 
   39| from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound
   40| 
   46| HORIZON_ATTENTION_GAIN_INIT = 1.0e-2
   47| 
   60| PERSISTENCE_DECAY_HALFLIFE = 5.0
   61| 
   63| class _HorizonRefine(nn.Module):
   74| 
   75|     def __init__(
   76|         self,
   77|         d_hidden: int,
   78|         kernel_size: int = 3,
   79|         dilations: Tuple[int, ...] = (1, 2),
   80|         film_cond_dim: Optional[int] = None,
   81|     ) -> None:
   94|         super().__init__()
   95|         self.blocks = nn.ModuleList()
   96|         for dilation in dilations:
   99|             padding = (kernel_size // 2) * dilation
  100|             self.blocks.append(
  101|                 nn.ModuleDict(
  102|                     {
  103|                         "conv": nn.Conv1d(
  104|                             d_hidden,
  105|                             d_hidden,
  106|                             kernel_size=kernel_size,
  107|                             padding=padding,
  108|                             dilation=dilation,
  109|                         ),
  110|                         "norm": nn.GroupNorm(
  111|                             num_groups=min(8, d_hidden), num_channels=d_hidden
  112|                         ),
  113|                     }
  114|                 )
  115|             )
  116| 
  120|         self.film: Optional[nn.ModuleList]
  121|         if film_cond_dim is not None:
  122|             self.film = nn.ModuleList(
  123|                 [nn.Linear(film_cond_dim, 2 * d_hidden) for _ in dilations]
  124|             )
  125|             for layer in self.film:
  126|                 layer = cast(nn.Linear, layer)
  127|                 nn.init.zeros_(layer.weight)
  128|                 nn.init.zeros_(layer.bias)
  129|         else:
  130|             self.film = None
  131| 
  132|     def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
  143|         for index, block in enumerate(self.blocks):
  144|             block = cast(nn.ModuleDict, block)
  145|             y = block["conv"](x)
  146|             y = F.gelu(block["norm"](y))
  147|             if self.film is not None and cond is not None:
  148|                 film = cast(nn.ModuleList, self.film)
  149|                 gamma, beta = cast(nn.Linear, film[index])(cond).chunk(2, dim=-1)
  151|                 y = y * (1.0 + gamma[..., None]) + beta[..., None]
  152|             x = x + y
  153|         return x
  154| 
  156| class _HorizonSelfAttention(nn.Module):
  180| 
  181|     def __init__(self, d_hidden: int, num_heads: int) -> None:
  189|         super().__init__()
  190|         self.num_heads = int(num_heads)
  191|         self.d_head = d_hidden // self.num_heads
  192| 
  193|         self.norm = nn.LayerNorm(d_hidden)
  196|         self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)
  197|         self.k_proj = nn.Linear(d_hidden, d_hidden, bias=False)
  198|         self.v_proj = nn.Linear(d_hidden, d_hidden, bias=False)
  199|         self.out_proj = nn.Linear(d_hidden, d_hidden, bias=False)
  200| 
  206|         self.residual_gain = nn.Parameter(torch.full((1,), HORIZON_ATTENTION_GAIN_INIT))
  207| 
  208|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  218|         n_rows, horizon, d_hidden = x.shape
  219|         normed = self.norm(x)
  221|         query = self.q_proj(normed).reshape(n_rows, horizon, self.num_heads, self.d_head)
  222|         key = self.k_proj(normed).reshape(n_rows, horizon, self.num_heads, self.d_head)
  223|         value = self.v_proj(normed).reshape(n_rows, horizon, self.num_heads, self.d_head)
  224|         attended = F.scaled_dot_product_attention(
  225|             query.transpose(1, 2),
  226|             key.transpose(1, 2),
  227|             value.transpose(1, 2),
  230|             dropout_p=0.0,
  231|         )
  232|         attended = attended.transpose(1, 2).reshape(n_rows, horizon, d_hidden)
  233|         return x + self.residual_gain * self.out_proj(attended)
  234| 
  236| class HorizonDecoderCore(nn.Module):
  254| 
  255|     def __init__(
  256|         self,
  257|         d_hidden: int = 128,
  258|         horizon: int = 30,
  259|         kernel_size: int = 3,
  260|         depth: int = 2,
  261|         film: bool = False,
  262|         film_per_block: bool = False,
  263|         attention_blocks: int = 0,
  264|         attention_heads: int = 4,
  265|     ) -> None:
  293|         super().__init__()
  294|         self.horizon = int(horizon)
  295|         self.d_hidden = int(d_hidden)
  296|         self.film = bool(film)
  297|         self.film_per_block = bool(film_per_block)
  298|         if self.film_per_block and not self.film:
  299|             raise ValueError(
  300|                 "film_per_block=True requires film=True; per-block FiLM is a form of FiLM, not a "
  301|                 "separate mechanism"
  302|             )
  303| 
  306|         self.horizon_embedding = nn.Parameter(torch.zeros(horizon, d_hidden))
  307|         nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
  308| 
  309|         depth = max(1, int(depth))
  310|         dilations = tuple(2**index for index in range(depth))
  311|         refine_film_dim = d_hidden if (self.film and self.film_per_block) else None
  312|         self.refine = _HorizonRefine(
  313|             d_hidden, kernel_size=kernel_size, dilations=dilations, film_cond_dim=refine_film_dim
  314|         )
  315| 
  316|         if self.film and not self.film_per_block:
  320|             self.film_gen: Optional[nn.Linear] = nn.Linear(d_hidden, 2 * d_hidden)
  321|             nn.init.zeros_(self.film_gen.weight)
  322|             nn.init.zeros_(self.film_gen.bias)
  323|         else:
  324|             self.film_gen = None
  325| 
  329|         self.attention_blocks = max(0, int(attention_blocks))
  330|         self.attention_heads = int(attention_heads)
  331|         self.attention: Optional[nn.ModuleList]
  332|         if self.attention_blocks > 0:
  333|             if self.attention_heads < 1 or self.d_hidden % self.attention_heads != 0:
  334|                 raise ValueError(
  335|                     f"attention_heads={self.attention_heads} must be a positive divisor of "
  336|                     f"d_hidden={self.d_hidden}; the horizon tokens are split evenly across heads "
  337|                     f"and a remainder would silently drop channels"
  338|                 )
  339|             self.attention = nn.ModuleList(
  340|                 _HorizonSelfAttention(self.d_hidden, self.attention_heads)
  341|                 for _ in range(self.attention_blocks)
  342|             )
  343|         else:
  344|             self.attention = None
  345| 
  346|         self.out_norm = nn.LayerNorm(d_hidden)
  347| 
  348|     def decode(self, h: torch.Tensor) -> torch.Tensor:
  357|         batch, seq_len, d_hidden = h.shape
  358|         horizon = self.horizon
  359| 
  360|         feat = h.unsqueeze(2).expand(-1, -1, horizon, -1)
  361|         feat = feat + self.horizon_embedding[None, None, :, :]
  362|         if self.film_gen is not None:
  364|             gamma, beta = self.film_gen(h).chunk(2, dim=-1)
  365|             feat = feat * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]
  366| 
  367|         skip = feat
  369|         flat = feat.reshape(batch * seq_len, horizon, d_hidden).transpose(1, 2).contiguous()
  372|         cond = h.reshape(batch * seq_len, d_hidden) if self.film_per_block else None
  373|         flat = self.refine(flat, cond)
  374| 
  379|         feat = flat.transpose(1, 2)
  380|         if self.attention is not None:
  381|             for block in self.attention:
  382|                 feat = block(feat)
  383| 
  384|         feat = feat.reshape(batch, seq_len, horizon, d_hidden)
  385|         return self.out_norm(feat + skip)
  386| 
  388| class BaselineFutureDecoder(nn.Module):
  395| 
  399|     persistence_weight: Optional[nn.Parameter]
  400| 
  401|     def __init__(
  402|         self,
  403|         core: HorizonDecoderCore,
  404|         d_model: int = 128,
  405|         out_channels: int = 109,
  406|         d_hidden: int = 128,
  407|         dropout: float = 0.1,
  408|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  409|         persistence_residual: bool = False,
  410|     ) -> None:
  429|         super().__init__()
  430|         self.core = core
  431|         self.out_channels = int(out_channels)
  432|         self.d_hidden = int(d_hidden)
  433|         self.logvar_clamp = logvar_clamp
  434| 
  435|         self.proj = ResidualMLP(
  436|             input_dim=d_model,
  437|             hidden_dims=geometric_schedule(d_model, d_hidden, 3),
  438|             final_activation=True,
  439|             use_skip_connection=True,
  440|             use_input_layer_norm=True,
  441|             activation=nn.GELU,
  442|             dropout=dropout,
  443|         )
  444|         self.mean_head = nn.Linear(d_hidden, out_channels)
  445|         self.logvar_head = nn.Linear(d_hidden, out_channels)
  446| 
  455|         if persistence_residual:
  456|             steps = torch.arange(self.core.horizon, dtype=torch.float32)
  457|             decay = torch.pow(0.5, steps / PERSISTENCE_DECAY_HALFLIFE)
  458|             self.persistence_weight = nn.Parameter(
  459|                 decay[:, None].repeat(1, self.out_channels)
  460|             )
  461|         else:
  462|             self.persistence_weight = None
  463| 
  464|     def forward(
  465|         self,
  466|         decoder_state: torch.Tensor,
  467|         persistence: Optional[torch.Tensor] = None,
  468|     ) -> Tuple[torch.Tensor, torch.Tensor]:
  490|         if (self.persistence_weight is None) != (persistence is None):
  491|             built = "with" if self.persistence_weight is not None else "without"
  492|             called = "with" if persistence is not None else "without"
  493|             raise ValueError(
  494|                 f"the decoder was built {built} a persistence residual and was called {called} a "
  495|                 f"persistence input. The two are one decision: persistence_residual builds the "
  496|                 f"weight and the forward that supplies it, and half of that is a model whose mean "
  497|                 f"forecast is not the one its configuration describes."
  498|             )
  499|         h = self.proj(decoder_state)
  500|         feat = self.core.decode(h)
  501|         mu_base = self.mean_head(feat)
  502|         if self.persistence_weight is not None and persistence is not None:
  504|             mu_base = mu_base + self.persistence_weight * persistence[..., None, :]
  505|         logvar_base = smooth_bound(self.logvar_head(feat), *self.logvar_clamp)
  506|         return mu_base, logvar_base
  507| 
  509| class ResidualFutureDecoder(nn.Module):
  520| 
  521|     def __init__(
  522|         self,
  523|         core: HorizonDecoderCore,
  524|         d_model: int = 128,
  525|         d_z: int = 24,
  526|         out_channels: int = 109,
  527|         d_hidden: int = 128,
  528|         dropout: float = 0.1,
  529|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  530|     ) -> None:
  542|         super().__init__()
  543|         self.core = core
  544|         self.out_channels = int(out_channels)
  545|         self.d_hidden = int(d_hidden)
  546|         self.logvar_clamp = logvar_clamp
  547| 
  548|         in_dim = d_model + d_z
  549|         self.proj = ResidualMLP(
  550|             input_dim=in_dim,
  551|             hidden_dims=geometric_schedule(in_dim, d_hidden, 3),
  552|             final_activation=True,
  553|             use_skip_connection=True,
  554|             use_input_layer_norm=True,
  555|             activation=nn.GELU,
  556|             dropout=dropout,
  557|         )
  558|         self.mean_head = nn.Linear(d_hidden, out_channels)
  559|         self.logvar_head = nn.Linear(d_hidden, out_channels)
  560| 
  561|     def forward(
  562|         self, decoder_state: torch.Tensor, z: torch.Tensor
  563|     ) -> Tuple[torch.Tensor, torch.Tensor]:
  573|         h = self.proj(torch.cat([decoder_state, z], dim=-1))
  574|         feat = self.core.decode(h)
  575|         delta_mu_src = self.mean_head(feat)
  576|         logvar_full = smooth_bound(self.logvar_head(feat), *self.logvar_clamp)
  577|         return delta_mu_src, logvar_full
```

## 8. Geometry — `teb_vae/lag_attn_rws/nets/geometry.py`

`teb_vae/lag_attn_rws/nets/geometry.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   32| from __future__ import annotations
   33| 
   34| from dataclasses import dataclass
   35| 
   37| @dataclass(frozen=True)
   38| class TrimmedRawGeometry:
   51| 
   52|     raw_len: int
   53|     decimation: int
   54|     horizon: int
   55|     warmup: int
   56| 
   57|     def __post_init__(self) -> None:
   65|         if self.decimation < 1:
   66|             raise ValueError(f"decimation must be >= 1, got {self.decimation}")
   67|         if self.raw_len % self.decimation != 0:
   68|             raise ValueError(
   69|                 f"raw_len ({self.raw_len}) must be divisible by decimation "
   70|                 f"({self.decimation}); a trimmed segment holds a whole number of steps"
   71|             )
   72|         if self.horizon < 1:
   73|             raise ValueError(f"horizon must be >= 1, got {self.horizon}")
   74|         if self.t_valid < 1:
   75|             raise ValueError(
   76|                 f"degenerate geometry: T={self.t} with horizon={self.horizon} leaves "
   77|                 f"T_valid={self.t_valid} anchors with a fully observed forecast window"
   78|             )
   79|         if not 0 <= self.warmup < self.t_valid:
   80|             raise ValueError(
   81|                 f"warmup ({self.warmup}) must satisfy 0 <= warmup < T - H "
   82|                 f"({self.t_valid}); otherwise no anchor survives the warm-up mask"
   83|             )
   84| 
   88|         assert self.future_block_start(0) == self.decimation
   89|         assert self.n_raw(0) == self.decimation - 1
   90|         assert self.future_block_start(0) == self.n_raw(0) + 1
   91|         last_anchor = self.t_valid - 1
   92|         last_target_end = self.future_block_start(last_anchor) + self.horizon * self.r - 1
   93|         assert last_target_end == self.raw_len - 1
   94| 
   95|     @property
   96|     def t(self) -> int:
   98|         return self.raw_len // self.decimation
   99| 
  100|     @property
  101|     def t_valid(self) -> int:
  103|         return self.t - self.horizon
  104| 
  105|     @property
  106|     def r(self) -> int:
  108|         return self.decimation
  109| 
  110|     def n_raw(self, t: int) -> int:
  119|         return self.decimation * (t + 1) - 1
  120| 
  121|     def future_block_start(self, t: int) -> int:
  130|         return self.n_raw(t) + 1
  131| 
  132|     def valid_anchor_range(self) -> range:
  134|         return range(self.warmup, self.t_valid)
  135| 
  137| if __name__ == "__main__":
  140|     geometry = TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30)
  141|     first, last = 0, geometry.t_valid - 1
  142|     print("trimmed-grid geometry (raw_len=4800, decimation=16, horizon=30, warmup=30)")
  143|     print(f"  T = {geometry.t}, T_valid = {geometry.t_valid}, R = {geometry.r}")
  144|     print(f"  trained anchors: [{geometry.valid_anchor_range().start}, "
  145|           f"{geometry.valid_anchor_range().stop})")
  146|     for t in (first, last):
  147|         start = geometry.future_block_start(t)
  148|         stop = start + geometry.horizon * geometry.r
  149|         print(f"  anchor {t:>3}: n_raw = {geometry.n_raw(t):>4}, forecast [{start}, {stop})")
  150|     print(f"  contrast: an untrimmed grid with an in-model crop of 15 starts anchor 0's "
  151|           f"forecast at 16*(0+16) = {16 * 16}, one minute later")
```

## 9. Base architecture, conv-LSTM cells — `teb_vae/lag_attn_rws/nets/model.py` (`SeqVaeLagAttnRws`)

Constructor, hooks (`_default_decoder_out_channels`, `_build_channel_gate`, `_build_adapter`, `source_kv_body/_modules`, `encode_source_kv`, `build_lag_mask`, `_prior_clock_dim`, `_check_persistence_target`), init policy, `_reparameterize_shared`, the **dense** `forward` (overridden by `CausalWarmupInputs.forward` in all four causal models), `kld_tensor`, `compute_loss` delegation.

`teb_vae/lag_attn_rws/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   46| from __future__ import annotations
   47| 
   48| import math
   49| from typing import Any, Dict, Optional, Sequence, Tuple, cast
   50| 
   51| import torch
   52| from torch import nn
   53| 
   54| from teb_vae.lag_attn.nets.attention import LagCrossAttention
   55| from teb_vae.lag_attn.nets.blocks import causalize_norms, initialization, validate_choice
   56| from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
   57| from teb_vae.lag_attn.nets.encoders import (
   58|     LAG_KV_SOURCE_CHOICES,
   59|     AvailabilityInputAdapter,
   60|     CausalConvLstmEncoder,
   61|     CausalConvStem,
   62| )
   63| from teb_vae.lag_attn.nets.heads import POSTERIOR_LOGVAR_MODES, PosteriorHead, TEAnalysisHead
   64| from teb_vae.lag_attn.nets.delays import ChannelGate
   65| from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
   66| from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
   67| from teb_vae.lag_attn_rws.nets.losses import (
   68|     LOGVAR_FLOOR_MARGIN_FRAC,  # noqa: F401  re-exported; every consumer imports it from here
   69| )
   70| from teb_vae.lag_attn_rws.nets.losses import (
   71|     compute_loss as compute_raw_objective,
   72| )
   73| from teb_vae.lag_attn_rws.nets.losses import horizon_decay_weight
   74| from teb_vae.lag_attn_rws.nets.losses import (
   75|     kld_tensor as closed_form_kld,
   76| )
   77| from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index, build_future_target
   78| 
   82| SATURATION_FRAC = 0.99
   83| 
   88| BASE_DECODE_CHOICES = ("sample", "mean")
   89| 
   91| class SeqVaeLagAttnRws(nn.Module):
   98| 
  101|     target_gate: Optional[ChannelGate]
  102|     source_gate: Optional[ChannelGate]
  103| 
  106|     future_index: torch.Tensor
  107| 
  116|     horizon_weight: Optional[torch.Tensor]
  117| 
  118|     def __init__(
  119|         self,
  120|         *,
  121|         sequence_length: int = 300,
  122|         d_model: int = 128,
  123|         d_z: int = 48,
  124|         horizon: int = 30,
  125|         raw_per_step: int = 16,
  126|         warmup_period: int = 30,
  127|         c_y: int = 109,
  128|         c_u: int = 58,
  129|         use_up_st: bool = True,
  130|         max_lag: int = 90,
  131|         num_heads: int = 4,
  132|         d_head: int = 32,
  133|         lstm_layers: int = 2,
  134|         dropout: float = 0.1,
  135|         decoder_hidden: int = 128,
  136|         decoder_out_channels: Optional[int] = None,
  137|         horizon_depth: int = 2,
  138|         horizon_kernel: int = 3,
  139|         horizon_film: bool = False,
  140|         horizon_attention_blocks: int = 0,
  141|         horizon_embed_std: float = 0.02,
  142|         head_init_calibration: bool = False,
  143|         a_head_gain: float = 1.0,
  144|         encoder_extra_dilations: Tuple[int, ...] = (),
  145|         encoder_extra_kernel: int = 15,
  146|         conv_norm_groups: Optional[int] = None,
  147|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  148|         mu_scale: float = 5.0,
  149|         delta_mu_scale: float = 3.0,
  150|         delta_logvar_scale: float = 2.0,
  151|         posterior_logvar_mode: str = "residual",
  152|         source_dropout: Optional[float] = None,
  153|         lag_kv_source: str = "encoder",
  154|         use_entmax: bool = False,
  155|         attention_grad_checkpoint: bool = False,
  156|         lag_bias_init: str = "normal",
  157|         alibi_slope_scale: float = 1.0,
  158|         query_uses_logvar: bool = False,
  159|         prior_availability_input: bool = False,
  160|         causal_norm: bool = False,
  161|         coverage_floor: float = 0.9,
  162|         base_decode: str = "sample",
  163|         persistence_residual: bool = False,
  164|         horizon_weight_halflife_steps: Optional[float] = None,
  165|         target_keep_index: Optional[Sequence[int]] = None,
  166|         target_delays: Optional[Sequence[int]] = None,
  167|         source_keep_index: Optional[Sequence[int]] = None,
  168|         source_delays: Optional[Sequence[int]] = None,
  169|         init_weights: bool = True,
  170|     ) -> None:
  350|         super().__init__()
  351| 
  357|         if int(c_y) < 1 or int(c_u) < 1:
  358|             raise ValueError(
  359|                 f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
  360|             )
  361|         if int(num_heads) * int(d_head) != int(d_model):
  362|             raise ValueError(
  363|                 f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
  364|             )
  368|         if int(max_lag) < 0:
  369|             raise ValueError(f"max_lag must be >= 0, got {max_lag}")
  372|         if int(d_z) % int(num_heads) != 0:
  373|             raise ValueError(
  374|                 f"the head-structured latent requires d_z % num_heads == 0, "
  375|                 f"got d_z={d_z}, num_heads={num_heads}"
  376|             )
  377| 
  381|         self.geometry = TrimmedRawGeometry(
  382|             raw_len=int(sequence_length) * int(raw_per_step),
  383|             decimation=int(raw_per_step),
  384|             horizon=int(horizon),
  385|             warmup=int(warmup_period),
  386|         )
  387| 
  388|         self.sequence_length = int(sequence_length)
  389|         self.d_model = int(d_model)
  390|         self.d_z = int(d_z)
  391|         self.horizon = int(horizon)
  392|         self.raw_per_step = int(raw_per_step)
  393|         self.warmup_period = int(warmup_period)
  394|         self.c_y = int(c_y)
  395|         self.c_u = int(c_u)
  396|         self.use_up_st = bool(use_up_st)
  397|         self.max_lag = int(max_lag)
  398|         self.num_heads = int(num_heads)
  399|         self.mu_scale = float(mu_scale)
  400|         self.delta_mu_scale = float(delta_mu_scale)
  401|         self.delta_logvar_scale = float(delta_logvar_scale)
  402|         self.posterior_logvar_mode = validate_choice(
  403|             posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
  404|         )
  417|         self.source_dropout = float(dropout if source_dropout is None else source_dropout)
  418|         self.posterior_source_dropout = 0.0 if source_dropout is None else float(source_dropout)
  419|         self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
  420|         self.coverage_floor = float(coverage_floor)
  421|         self.base_decode = validate_choice(base_decode, BASE_DECODE_CHOICES, "base_decode")
  426|         self.persistence_residual = bool(persistence_residual)
  427|         if self.persistence_residual:
  428|             self._check_persistence_target()
  432|         self.horizon_embed_std = float(horizon_embed_std)
  433|         self.head_init_calibration = bool(head_init_calibration)
  434|         self.a_head_gain = float(a_head_gain)
  435| 
  440|         self.register_buffer(
  441|             "future_index", build_future_index(self.geometry), persistent=False
  442|         )
  443| 
  456|         self.horizon_weight_halflife_steps = (
  457|             None
  458|             if horizon_weight_halflife_steps is None
  459|             else float(horizon_weight_halflife_steps)
  460|         )
  461|         if self.horizon_weight_halflife_steps is not None:
  462|             self.register_buffer(
  463|                 "horizon_weight",
  464|                 horizon_decay_weight(self.horizon_weight_halflife_steps, self.horizon),
  465|                 persistent=False,
  466|             )
  467| 
  472|         self.target_gate = self._build_channel_gate(
  473|             self.c_y, target_keep_index, target_delays
  474|         )
  475|         self.source_gate = self._build_channel_gate(
  476|             self.c_u, source_keep_index, source_delays
  477|         )
  497|         self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)
  498|         self.source_adapter = self._build_adapter(
  499|             self.source_gate, self.c_u, self.source_dropout
  500|         )
  501| 
  511|         extra_dilations = tuple(int(x) for x in encoder_extra_dilations)
  512|         extra_kernels = tuple(int(encoder_extra_kernel) for _ in extra_dilations)
  513|         encoder_dilations = (1, 2, 4) + extra_dilations
  514|         self.target_encoder = CausalConvLstmEncoder(
  515|             d_model=d_model,
  516|             cnn_kernels=(3, 7, 11) + extra_kernels,
  517|             cnn_dilations=encoder_dilations,
  518|             lstm_layers=lstm_layers,
  519|             lstm_dropout=dropout,
  520|             conv_dropout=dropout,
  521|             stack_skip_connection=False,
  522|             post_residual_activation=False,
  523|             conv_norm_groups=conv_norm_groups,
  524|         )
  529|         self.lag_kv_source = validate_choice(
  530|             lag_kv_source, LAG_KV_SOURCE_CHOICES, "lag_kv_source"
  531|         )
  532|         if self.lag_kv_source == "encoder":
  533|             self.source_encoder = CausalConvLstmEncoder(
  534|                 d_model=d_model,
  535|                 cnn_kernels=(3, 5, 11) + extra_kernels,
  536|                 cnn_dilations=encoder_dilations,
  537|                 lstm_layers=lstm_layers,
  538|                 lstm_dropout=self.source_dropout,
  539|                 conv_dropout=self.source_dropout,
  540|                 stack_skip_connection=False,
  541|                 post_residual_activation=False,
  542|                 conv_norm_groups=conv_norm_groups,
  543|             )
  544|         elif self.lag_kv_source == "conv_stem":
  554|             self.source_kv_stem = CausalConvStem(
  555|                 d_model=d_model,
  556|                 cnn_kernels=(3, 5, 11) + extra_kernels,
  557|                 cnn_dilations=encoder_dilations,
  558|                 conv_dropout=self.source_dropout,
  559|                 conv_norm_groups=conv_norm_groups,
  560|             )
  561| 
  565|         self.prior_availability_input = bool(prior_availability_input)
  566|         self.prior_head = FullLatentPriorHead(
  567|             d_model=d_model,
  568|             d_z=d_z,
  569|             logvar_clamp=logvar_clamp,
  570|             dropout=dropout,
  571|             mu_scale=self.mu_scale,
  572|             clock_dim=self._prior_clock_dim() if self.prior_availability_input else None,
  573|         )
  574| 
  581|         self.query_uses_logvar = bool(query_uses_logvar)
  582|         query_in = 2 * self.d_z if self.query_uses_logvar else self.d_z
  583|         self.query_proj = nn.Linear(query_in, d_model)
  584| 
  588|         self.lag_attn = LagCrossAttention(
  589|             d_model=d_model,
  590|             num_heads=num_heads,
  591|             d_head=d_head,
  592|             max_lag=max_lag,
  593|             dropout=0.0,
  594|             use_entmax=use_entmax,
  595|             grad_checkpoint=attention_grad_checkpoint,
  596|             lag_bias_init=lag_bias_init,
  597|             alibi_slope_scale=alibi_slope_scale,
  598|         )
  599|         self.posterior_head = PosteriorHead(
  600|             d_model=d_model,
  601|             d_z=d_z,
  602|             logvar_clamp=logvar_clamp,
  603|             dropout=dropout,
  604|             delta_mu_scale=self.delta_mu_scale,
  605|             head_structured=True,
  606|             num_heads=num_heads,
  607|             d_head=d_head,
  608|             delta_logvar_scale=self.delta_logvar_scale,
  609|             posterior_logvar_mode=self.posterior_logvar_mode,
  612|             source_dropout=self.posterior_source_dropout,
  613|         )
  614|         self.te_analysis = TEAnalysisHead()
  615| 
  633|         self.horizon_core = HorizonDecoderCore(
  634|             d_hidden=decoder_hidden,
  635|             horizon=horizon,
  636|             kernel_size=horizon_kernel,
  637|             depth=horizon_depth,
  638|             film=horizon_film,
  639|             film_per_block=True,
  640|             attention_blocks=horizon_attention_blocks,
  641|         )
  642|         self.decoder_out_channels = (
  643|             self._default_decoder_out_channels()
  644|             if decoder_out_channels is None
  645|             else int(decoder_out_channels)
  646|         )
  647|         self.decoder = BaselineFutureDecoder(
  648|             core=self.horizon_core,
  649|             d_model=d_z,
  650|             out_channels=self.decoder_out_channels,
  651|             d_hidden=decoder_hidden,
  652|             dropout=0.0,
  653|             logvar_clamp=logvar_clamp,
  656|             persistence_residual=self.persistence_residual,
  657|         )
  658| 
  659|         self.causal_norm = bool(causal_norm)
  660|         if self.causal_norm:
  665|             source_body = self.source_kv_body()
  666|             self.n_causalized_norms = causalize_norms(self.target_encoder) + (
  667|                 0 if source_body is None else causalize_norms(source_body)
  668|             )
  669|         else:
  670|             self.n_causalized_norms = 0
  671| 
  675|         for parameter in self.lag_attn.W_o.parameters():
  676|             parameter.requires_grad_(False)
  677| 
  678|         if init_weights:
  679|             initialization(self)
  682|         self._zero_init_delta_heads()
  687|         self.prior_head.zero_init_clock()
  694|         self._zero_init_film_generators()
  695| 
  700|         if self.horizon_embed_std != 0.02:
  701|             self._reinit_horizon_embedding()
  702|         if self.head_init_calibration:
  703|             self._calibrate_output_heads()
  704|             self._calibrate_prior_scale()
  705|         if self.a_head_gain != 1.0:
  706|             self._set_a_head_gain()
  707| 
  708|     def _default_decoder_out_channels(self) -> int:
  727|         return self.raw_per_step
  728| 
  729|     @staticmethod
  730|     def _build_channel_gate(
  731|         declared_width: int,
  732|         keep_index: Optional[Sequence[int]],
  733|         delays: Optional[Sequence[int]],
  734|     ) -> Optional[ChannelGate]:
  754|         if keep_index is None and delays is None:
  755|             return None
  756|         return ChannelGate(
  757|             declared_width=int(declared_width), keep_index=keep_index, delays=delays
  758|         )
  759| 
  760|     def _build_adapter(
  761|         self, gate: Optional[ChannelGate], declared_width: int, dropout: float
  762|     ) -> AvailabilityInputAdapter:
  779|         width = declared_width if gate is None else gate.out_channels
  780|         delays = None if gate is None else [int(value) for value in gate.delay.delay_steps]
  781|         return AvailabilityInputAdapter(
  782|             in_dim=width,
  783|             d_model=self.d_model,
  784|             sequence_length=self.sequence_length,
  785|             dropout=dropout,
  786|             delays=delays,
  787|         )
  788| 
  789|     def source_kv_body(self) -> Optional[nn.Module]:
  801|         if self.lag_kv_source == "encoder":
  802|             return self.source_encoder
  803|         if self.lag_kv_source == "conv_stem":
  804|             return self.source_kv_stem
  805|         return None
  806| 
  807|     def source_kv_modules(self) -> Tuple[nn.Module, ...]:
  818|         body = self.source_kv_body()
  819|         return (self.source_adapter,) if body is None else (self.source_adapter, body)
  820| 
  821|     def encode_source_kv(self, source: torch.Tensor) -> torch.Tensor:
  832|         encoded = source
  833|         for module in self.source_kv_modules():
  834|             encoded = module(encoded)
  835|         return encoded
  836| 
  837|     def build_lag_mask(
  838|         self, seq_len: int, device: Optional[torch.device] = None
  839|     ) -> torch.Tensor:
  856|         return self.lag_attn.build_lag_mask(seq_len, device=device)
  857| 
  858|     @property
  859|     def source_delay_steps(self) -> int:
  884|         return 0 if self.source_gate is None else self.source_gate.max_delay
  885| 
  886|     @staticmethod
  887|     def _zero_linear(layer: nn.Linear) -> None:
  889|         nn.init.zeros_(layer.weight)
  890|         if layer.bias is not None:
  891|             nn.init.zeros_(layer.bias)
  892| 
  893|     def _prior_clock_dim(self) -> int:
  915|         raise ValueError(
  916|             f"prior_availability_input=True on {type(self).__name__}, whose input streams carry no "
  917|             f"warm-up: every channel is honest at every step, so nothing arrives late and there is "
  918|             f"no availability term in the KL to cancel. The flag belongs to the causal cells, whose "
  919|             f"one-sided inputs arrive over the first steps of a segment."
  920|         )
  921| 
  922|     def _check_persistence_target(self) -> None:
  944|         raise ValueError(
  945|             f"persistence_residual=True on {type(self).__name__}, whose forecast block is R raw "
  946|             f"samples of one signal per horizon token: there is no per-channel level for a "
  947|             f"persistence term to carry forward. The flag belongs to the feature-target cells, "
  948|             f"whose block's last axis counts stored coefficients of the target itself."
  949|         )
  950| 
  951|     def _zero_init_delta_heads(self) -> None:
  962|         for module in (self.posterior_head.delta_mu_head, self.posterior_head.delta_logvar_head):
  965|             if module is None:
  966|                 continue
  967|             layers = list(module) if isinstance(module, nn.ModuleList) else [module]
  968|             for layer in layers:
  969|                 self._zero_linear(cast(nn.Linear, layer))
  970| 
  978|         independent = self.posterior_head.logvar_post_head
  979|         if independent is not None:
  982|             lo, hi = self.posterior_head.logvar_clamp
  983|             if not lo < 0.0 < hi:
  984|                 raise ValueError(
  985|                     f"posterior_logvar_mode='independent' seeds the head at unit scale, which "
  986|                     f"needs 0 inside logvar_clamp; got ({lo}, {hi})"
  987|                 )
  988|             bias_value = math.log((0.0 - lo) / (hi - 0.0))
  989|             layers = (
  990|                 list(independent)
  991|                 if isinstance(independent, nn.ModuleList)
  992|                 else [independent]
  993|             )
  994|             for layer in layers:
  995|                 linear = cast(nn.Linear, layer)
  996|                 nn.init.zeros_(linear.weight)
  997|                 linear.bias.data.fill_(bias_value)
  998| 
  999|     def _zero_init_film_generators(self) -> None:
 1012|         core = self.horizon_core
 1013|         film_layers: list[nn.Module] = []
 1014|         if core.film_gen is not None:
 1015|             film_layers.append(core.film_gen)
 1016|         if core.refine.film is not None:
 1017|             film_layers.extend(core.refine.film)
 1018|         for layer in film_layers:
 1019|             self._zero_linear(cast(nn.Linear, layer))
 1020| 
 1021|     def _reinit_horizon_embedding(self) -> None:
 1032|         nn.init.normal_(
 1033|             self.horizon_core.horizon_embedding, mean=0.0, std=self.horizon_embed_std
 1034|         )
 1035| 
 1036|     def _calibrate_output_heads(self) -> None:
 1052|         self.decoder.mean_head.weight.data.mul_(0.02)
 1053|         self.decoder.logvar_head.bias.data.fill_(math.log(5.0 / 3.0))
 1054|         self.decoder.logvar_head.weight.data.mul_(0.1)
 1055| 
 1056|     def _calibrate_prior_scale(self) -> None:
 1082|         lo, hi = self.logvar_clamp
 1083|         if not lo < 0.0 < hi:
 1084|             raise ValueError(
 1085|                 f"prior scale calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
 1086|             )
 1087|         head = self.prior_head.logvar_prior_head
 1088|         if not isinstance(head.skip_proj, nn.Linear):
 1089|             raise ValueError(
 1090|                 "prior scale calibration requires a projected skip on the log-variance head; "
 1091|                 "with d_model == d_z the skip is an identity and the output cannot be pinned"
 1092|             )
 1093|         self._zero_linear(head.skip_proj)
 1094|         final = cast(nn.Linear, head.body[-1])
 1095|         nn.init.zeros_(final.weight)
 1096|         final.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))
 1097| 
 1098|     def _set_a_head_gain(self) -> None:
 1110|         nn.init.constant_(self.posterior_head.a_head_norm.weight, self.a_head_gain)
 1111| 
 1112|     def _reparameterize_shared(
 1113|         self,
 1114|         mu_prior: torch.Tensor,
 1115|         logvar_prior: torch.Tensor,
 1116|         mu_post: torch.Tensor,
 1117|         logvar_post: torch.Tensor,
 1118|     ) -> Tuple[torch.Tensor, torch.Tensor]:
 1145|         epsilon = torch.randn_like(mu_prior)
 1146|         z_post = mu_post + epsilon * torch.exp(0.5 * logvar_post)
 1147|         if self.base_decode == "mean":
 1151|             return mu_prior, z_post
 1152|         z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
 1153|         return z_prior, z_post
 1154| 
 1155|     def forward(
 1156|         self,
 1157|         y_st: torch.Tensor,
 1158|         y_ph: torch.Tensor,
 1159|         u_stream: torch.Tensor,
 1160|     ) -> Dict[str, torch.Tensor]:
 1207|         target = torch.cat([y_st, y_ph], dim=-1)
 1208|         if self.target_gate is not None:
 1209|             target = self.target_gate(target)
 1210|         source = u_stream if self.source_gate is None else self.source_gate(u_stream)
 1211| 
 1212|         h_y = self.target_encoder(self.target_adapter(target))
 1213|         h_u = self.encode_source_kv(source)
 1214| 
 1215|         mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(h_y)
 1216| 
 1220|         query = (
 1221|             torch.cat([mu_prior, logvar_prior], dim=-1)
 1222|             if self.query_uses_logvar
 1223|             else mu_prior
 1224|         )
 1225|         _, alpha, attended_heads = self.lag_attn(
 1226|             self.query_proj(query), h_u, self.build_lag_mask(h_u.shape[1], h_u.device)
 1227|         )
 1228| 
 1229|         mu_post, logvar_post = self.posterior_head(
 1230|             h_y, attended_heads, mu_prior, raw_logvar_prior
 1231|         )
 1232|         z_prior, z_post = self._reparameterize_shared(
 1233|             mu_prior, logvar_prior, mu_post, logvar_post
 1234|         )
 1235| 
 1238|         with torch.no_grad():
 1239|             mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
 1240|             delta_mu_sat_frac = (
 1241|                 (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
 1242|             ).float().mean()
 1243| 
 1247|         t_valid = self.geometry.t_valid
 1248|         mu_base, logvar_base = self.decoder(z_prior[:, :t_valid])
 1249|         mu_full, logvar_full = self.decoder(z_post[:, :t_valid])
 1250| 
 1254|         kld_btd = self.kld_tensor(
 1255|             mu_prior=mu_prior,
 1256|             logvar_prior=logvar_prior,
 1257|             mu_post=mu_post,
 1258|             logvar_post=logvar_post,
 1259|         )
 1260|         kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
 1261|             kld_btd, alpha, head_structured=True
 1262|         )
 1263| 
 1264|         return {
 1265|             "mu_prior": mu_prior,
 1266|             "logvar_prior": logvar_prior,
 1267|             "raw_logvar_prior": raw_logvar_prior,
 1268|             "mu_post": mu_post,
 1269|             "logvar_post": logvar_post,
 1270|             "z_prior": z_prior,
 1271|             "z_post": z_post,
 1272|             "target_state": h_y,
 1273|             "source_state": h_u,
 1274|             "attended_source_heads": attended_heads,
 1275|             "attn_weights": alpha,
 1276|             "mu_base": mu_base,
 1277|             "logvar_base": logvar_base,
 1278|             "mu_full": mu_full,
 1279|             "logvar_full": logvar_full,
 1280|             "kld_per_t": kld_per_t,
 1281|             "kld_per_t_per_head": kld_per_t_per_head,
 1282|             "source_kl_lag_map": source_kl_lag_map,
 1283|             "mu_prior_sat_frac": mu_prior_sat_frac,
 1284|             "delta_mu_sat_frac": delta_mu_sat_frac,
 1285|         }
 1286| 
 1287|     def kld_tensor(
 1288|         self,
 1289|         mu_prior: torch.Tensor,
 1290|         logvar_prior: torch.Tensor,
 1291|         mu_post: torch.Tensor,
 1292|         logvar_post: torch.Tensor,
 1293|     ) -> torch.Tensor:
 1311|         return closed_form_kld(mu_prior, logvar_prior, mu_post, logvar_post)
 1312| 
 1313|     def compute_loss(
 1314|         self,
 1315|         forward_outputs: Dict[str, torch.Tensor],
 1316|         fhr_raw: torch.Tensor,
 1317|         *,
 1318|         weight: torch.Tensor,
 1319|         beta: float = 1.0,
 1320|         beta_prior: float = 0.0,
 1321|         lambda_full: float = 1.0,
 1322|         lambda_base: float = 1.0,
 1323|         likelihood: str = "gaussian_nll",
 1324|         free_bits: float = 0.0,
 1325|         lambda_ms: float = 0.0,
 1326|         lambda_deriv: float = 0.0,
 1327|         lambda_boundary: float = 0.0,
 1328|     ) -> Dict[str, Any]:
 1396|         return compute_raw_objective(
 1397|             forward_outputs,
 1398|             build_future_target(fhr_raw, self.geometry, future_index=self.future_index),
 1399|             weight=weight,
 1400|             geometry=self.geometry,
 1402|             block_width=self.geometry.r,
 1403|             coverage_floor=self.coverage_floor,
 1404|             logvar_clamp=self.logvar_clamp,
 1405|             beta=beta,
 1406|             beta_prior=beta_prior,
 1407|             lambda_full=lambda_full,
 1408|             lambda_base=lambda_base,
 1409|             likelihood=likelihood,
 1410|             free_bits=free_bits,
 1411|             lambda_ms=lambda_ms,
 1412|             lambda_deriv=lambda_deriv,
 1413|             lambda_boundary=lambda_boundary,
 1418|             horizon_weight=getattr(self, "horizon_weight", None),
 1419|         )
```

## 10. conv-Transformer cells — `teb_vae/lag_attn_transformer_rws/nets/`

### 10.1 `blocks.py` — `RMSNorm`, `LayerScale`, `SwiGLUFeedForward`, `RotaryPositionEncoding`, `CausalDepthwiseConv1d`, `init_depthwise_`, `GatedCausalConvBlock`, `build_causal_window_mask`, `CausalSelfAttention`, `CausalTransformerBlock`

`GatedCausalConvBlock`: $x + \gamma\odot\mathrm{Dropout}(W_{out}\,\mathrm{SiLU}(\mathrm{RMSNorm}(\mathrm{DWConv}(v\odot\sigma(g)))))$, $[v,g]=W_{in}\mathrm{RMSNorm}(x)$, RF $=(k-1)r+1$. `CausalTransformerBlock`: $x+\gamma_a\,\mathrm{MHSA}(\mathrm{RMSNorm}\,x)$ then $x+\gamma_f\,\mathrm{SwiGLU}(\mathrm{RMSNorm}\,x)$; attention = RoPE on q,k, SDPA with `is_causal=True` (no window) or boolean band mask (window). All projections bias-free; `LayerScale` init 0.01.

`teb_vae/lag_attn_transformer_rws/nets/blocks.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   27| from __future__ import annotations
   28| 
   29| import math
   30| from typing import Optional, Tuple
   31| 
   32| import torch
   33| import torch.nn.functional as F
   34| from torch import nn
   35| 
   38| RMS_NORM_EPS = 1e-5
   39| 
   42| LAYER_SCALE_INIT = 1e-2
   43| 
   46| ROPE_BASE = 10000.0
   47| 
   49| class RMSNorm(nn.Module):
   71| 
   72|     def __init__(self, dim: int, eps: float = RMS_NORM_EPS) -> None:
   79|         super().__init__()
   80|         self.dim = int(dim)
   81|         self.eps = float(eps)
   82|         self.weight = nn.Parameter(torch.ones(dim))
   83| 
   84|     def forward(self, x: torch.Tensor) -> torch.Tensor:
   93|         mean_square = x.pow(2).mean(dim=-1, keepdim=True)
   94|         return x * torch.rsqrt(mean_square + self.eps) * self.weight
   95| 
   96|     def extra_repr(self) -> str:
   98|         return f"{self.dim}, eps={self.eps}"
   99| 
  101| class LayerScale(nn.Module):
  116| 
  117|     def __init__(self, dim: int, init_value: float = LAYER_SCALE_INIT) -> None:
  124|         super().__init__()
  125|         self.dim = int(dim)
  126|         self.init_value = float(init_value)
  127|         self.weight = nn.Parameter(torch.full((dim,), float(init_value)))
  128| 
  129|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  138|         return x * self.weight
  139| 
  140|     def extra_repr(self) -> str:
  142|         return f"{self.dim}, init={self.init_value}"
  143| 
  145| class SwiGLUFeedForward(nn.Module):
  161| 
  162|     def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
  170|         super().__init__()
  171|         self.d_model = int(d_model)
  172|         self.d_ff = int(d_ff)
  173|         self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
  174|         self.value_proj = nn.Linear(d_model, d_ff, bias=False)
  175|         self.out_proj = nn.Linear(d_ff, d_model, bias=False)
  176|         self.dropout = nn.Dropout(dropout)
  177| 
  178|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  187|         gated = F.silu(self.gate_proj(x)) * self.value_proj(x)
  188|         return self.dropout(self.out_proj(gated))
  189| 
  191| class RotaryPositionEncoding(nn.Module):
  224| 
  227|     cos_table: torch.Tensor
  228|     sin_table: torch.Tensor
  229| 
  230|     def __init__(self, d_head: int, max_seq_len: int, base: float = ROPE_BASE) -> None:
  241|         super().__init__()
  242|         if d_head % 2 != 0:
  243|             raise ValueError(
  244|                 f"d_head must be even for rotary position encoding, which rotates coordinate "
  245|                 f"pairs; got d_head={d_head}"
  246|             )
  247|         if max_seq_len < 1:
  248|             raise ValueError(f"max_seq_len must be at least 1, got {max_seq_len}")
  249| 
  250|         self.d_head = int(d_head)
  251|         self.max_seq_len = int(max_seq_len)
  252|         self.base = float(base)
  253| 
  254|         exponents = torch.arange(0, d_head, 2, dtype=torch.float32) / float(d_head)
  255|         frequencies = torch.pow(torch.tensor(float(base)), -exponents)
  256|         angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), frequencies)
  257|         self.register_buffer("cos_table", angles.cos(), persistent=False)
  258|         self.register_buffer("sin_table", angles.sin(), persistent=False)
  259| 
  260|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  272|         seq_len = int(x.shape[-2])
  273|         if x.shape[-1] != self.d_head:
  274|             raise ValueError(
  275|                 f"expected a head width of {self.d_head}, got {int(x.shape[-1])}"
  276|             )
  277|         if seq_len > self.max_seq_len:
  278|             raise ValueError(
  279|                 f"sequence of {seq_len} steps exceeds the rotary tables built for "
  280|                 f"max_seq_len={self.max_seq_len}; the tables are fixed at construction so a "
  281|                 f"longer input cannot be served"
  282|             )
  283| 
  284|         cos = self.cos_table[:seq_len].to(dtype=x.dtype)
  285|         sin = self.sin_table[:seq_len].to(dtype=x.dtype)
  286|         pairs = x.reshape(*x.shape[:-1], self.d_head // 2, 2)
  287|         even, odd = pairs[..., 0], pairs[..., 1]
  288|         rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
  289|         return rotated.reshape(x.shape)
  290| 
  291|     def extra_repr(self) -> str:
  293|         return f"d_head={self.d_head}, max_seq_len={self.max_seq_len}, base={self.base}"
  294| 
  296| class CausalDepthwiseConv1d(nn.Module):
  308| 
  309|     def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
  320|         super().__init__()
  321|         if channels < 1 or kernel_size < 1 or dilation < 1:
  322|             raise ValueError(
  323|                 f"channels, kernel_size and dilation must all be positive; got "
  324|                 f"channels={channels}, kernel_size={kernel_size}, dilation={dilation}"
  325|             )
  326|         self.channels = int(channels)
  327|         self.kernel_size = int(kernel_size)
  328|         self.dilation = int(dilation)
  329|         self.left_padding = (int(kernel_size) - 1) * int(dilation)
  330|         self.conv = nn.Conv1d(
  331|             channels,
  332|             channels,
  333|             kernel_size=kernel_size,
  334|             groups=channels,
  335|             bias=False,
  336|             padding=0,
  337|             dilation=dilation,
  338|         )
  339| 
  340|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  349|         if self.left_padding > 0:
  350|             x = F.pad(x, (self.left_padding, 0))
  351|         return self.conv(x)
  352| 
  353|     def extra_repr(self) -> str:
  355|         return (
  356|             f"{self.channels}, kernel_size={self.kernel_size}, dilation={self.dilation}, "
  357|             f"left_padding={self.left_padding}"
  358|         )
  359| 
  361| def init_depthwise_(module: nn.Module) -> int:
  382|     count = 0
  383|     for child in module.modules():
  384|         if isinstance(child, CausalDepthwiseConv1d):
  385|             std = 1.0 / math.sqrt(float(child.kernel_size))
  386|             nn.init.normal_(child.conv.weight, mean=0.0, std=std)
  387|             count += 1
  388|     return count
  389| 
  391| class GatedCausalConvBlock(nn.Module):
  421| 
  422|     def __init__(
  423|         self,
  424|         d_model: int,
  425|         kernel_size: int,
  426|         dilation: int = 1,
  427|         dropout: float = 0.0,
  428|         layer_scale_init: float = LAYER_SCALE_INIT,
  429|     ) -> None:
  439|         super().__init__()
  440|         self.d_model = int(d_model)
  441|         self.norm_in = RMSNorm(d_model)
  442|         self.proj_in = nn.Linear(d_model, 2 * d_model, bias=False)
  443|         self.conv = CausalDepthwiseConv1d(d_model, kernel_size, dilation)
  444|         self.norm_conv = RMSNorm(d_model)
  445|         self.proj_out = nn.Linear(d_model, d_model, bias=False)
  446|         self.dropout = nn.Dropout(dropout)
  447|         self.layer_scale = LayerScale(d_model, layer_scale_init)
  448| 
  449|     @property
  450|     def receptive_field(self) -> int:
  452|         return self.conv.left_padding + 1
  453| 
  454|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  463|         value, gate = self.proj_in(self.norm_in(x)).chunk(2, dim=-1)
  464|         gated = value * torch.sigmoid(gate)
  465|         convolved = self.conv(gated.transpose(1, 2)).transpose(1, 2)
  466|         branch = self.proj_out(F.silu(self.norm_conv(convolved)))
  467|         return x + self.layer_scale(self.dropout(branch))
  468| 
  470| def build_causal_window_mask(seq_len: int, window: Optional[int]) -> torch.Tensor:
  493|     if seq_len < 1:
  494|         raise ValueError(f"seq_len must be at least 1, got {seq_len}")
  495|     if window is not None and window < 1:
  496|         raise ValueError(
  497|             f"attention window must be at least 1 step so that a row can attend to itself, "
  498|             f"got {window}"
  499|         )
  500| 
  501|     positions = torch.arange(seq_len)
  502|     displacement = positions[:, None] - positions[None, :]
  503|     allowed = displacement >= 0
  504|     if window is not None:
  505|         allowed = allowed & (displacement < int(window))
  506|     return allowed
  507| 
  509| class CausalSelfAttention(nn.Module):
  539| 
  542|     attn_mask: Optional[torch.Tensor]
  543| 
  544|     def __init__(
  545|         self,
  546|         d_model: int,
  547|         num_heads: int,
  548|         max_seq_len: int,
  549|         window: Optional[int] = None,
  550|         is_causal: Optional[bool] = None,
  551|         dropout: float = 0.0,
  552|         rope_base: float = ROPE_BASE,
  553|     ) -> None:
  573|         super().__init__()
  574|         if d_model % num_heads != 0:
  575|             raise ValueError(
  576|                 f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) so every head "
  577|                 f"has the same width"
  578|             )
  579|         derived_causal = window is None
  580|         if is_causal is not None and bool(is_causal) != derived_causal:
  581|             raise ValueError(
  582|                 f"is_causal={is_causal} contradicts window={window}: causality comes from the "
  583|                 f"kernel's is_causal flag when there is no window and from the explicit band mask "
  584|                 f"when there is one, never from both and never from neither"
  585|             )
  586| 
  587|         self.d_model = int(d_model)
  588|         self.num_heads = int(num_heads)
  589|         self.d_head = int(d_model) // int(num_heads)
  590|         self.max_seq_len = int(max_seq_len)
  591|         self.window = None if window is None else int(window)
  592|         self.is_causal = derived_causal
  593| 
  594|         self.norm = RMSNorm(d_model)
  595|         self.q_proj = nn.Linear(d_model, d_model, bias=False)
  596|         self.k_proj = nn.Linear(d_model, d_model, bias=False)
  597|         self.v_proj = nn.Linear(d_model, d_model, bias=False)
  598|         self.out_proj = nn.Linear(d_model, d_model, bias=False)
  600|         self.rope = RotaryPositionEncoding(self.d_head, max_seq_len, base=rope_base)
  601|         self.dropout = nn.Dropout(dropout)
  602| 
  603|         mask = None if self.window is None else build_causal_window_mask(max_seq_len, self.window)
  604|         self.register_buffer("attn_mask", mask, persistent=False)
  605| 
  606|     @property
  607|     def receptive_field(self) -> Optional[int]:
  609|         return self.window
  610| 
  611|     def _project(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  620|         batch, seq_len, _ = h.shape
  621|         shape = (batch, seq_len, self.num_heads, self.d_head)
  622|         query = self.q_proj(h).view(shape).transpose(1, 2)
  623|         key = self.k_proj(h).view(shape).transpose(1, 2)
  624|         value = self.v_proj(h).view(shape).transpose(1, 2)
  625|         return self.rope(query), self.rope(key), value
  626| 
  627|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  636|         batch, seq_len, _ = x.shape
  637|         query, key, value = self._project(self.norm(x))
  638|         mask = None if self.attn_mask is None else self.attn_mask[:seq_len, :seq_len]
  639|         attended = F.scaled_dot_product_attention(
  640|             query,
  641|             key,
  642|             value,
  643|             attn_mask=mask,
  644|             dropout_p=0.0,
  645|             is_causal=self.is_causal,
  646|         )
  647|         merged = attended.transpose(1, 2).reshape(batch, seq_len, self.d_model)
  648|         return self.dropout(self.out_proj(merged))
  649| 
  650|     def extra_repr(self) -> str:
  652|         context = "full causal prefix" if self.window is None else f"causal window {self.window}"
  653|         return f"d_model={self.d_model}, num_heads={self.num_heads}, context={context}"
  654| 
  656| class CausalTransformerBlock(nn.Module):
  680| 
  681|     def __init__(
  682|         self,
  683|         d_model: int,
  684|         num_heads: int,
  685|         d_ff: int,
  686|         max_seq_len: int,
  687|         window: Optional[int] = None,
  688|         dropout: float = 0.0,
  689|         layer_scale_init: float = LAYER_SCALE_INIT,
  690|         rope_base: float = ROPE_BASE,
  691|     ) -> None:
  704|         super().__init__()
  705|         self.attn = CausalSelfAttention(
  706|             d_model,
  707|             num_heads,
  708|             max_seq_len,
  709|             window=window,
  710|             dropout=dropout,
  711|             rope_base=rope_base,
  712|         )
  713|         self.attn_scale = LayerScale(d_model, layer_scale_init)
  714|         self.ffn_norm = RMSNorm(d_model)
  715|         self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout=dropout)
  716|         self.ffn_scale = LayerScale(d_model, layer_scale_init)
  717| 
  718|     @property
  719|     def window(self) -> Optional[int]:
  721|         return self.attn.window
  722| 
  723|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  732|         x = x + self.attn_scale(self.attn(x))
  733|         return x + self.ffn_scale(self.ffn(self.ffn_norm(x)))
```

### 10.2 `encoders.py` — `CausalConvTransformerEncoder`, `GatedCausalConvStem`

`teb_vae/lag_attn_transformer_rws/nets/encoders.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   24| from __future__ import annotations
   25| 
   26| from typing import Optional, Sequence
   27| 
   28| import torch
   29| from torch import nn
   30| 
   31| from teb_vae.lag_attn.nets.encoders import START_EMBED_STD, AvailabilityInputAdapter
   32| from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
   33| from teb_vae.lag_attn_transformer_rws.nets.blocks import (
   34|     LAYER_SCALE_INIT,
   35|     ROPE_BASE,
   36|     CausalTransformerBlock,
   37|     GatedCausalConvBlock,
   38|     RMSNorm,
   39| )
   40| 
   44| __all__ = [
   45|     "AvailabilityInputAdapter",
   46|     "CausalConvTransformerEncoder",
   47|     "GatedCausalConvStem",
   48|     "START_EMBED_STD",
   49|     "conv_receptive_field",
   50| ]
   51| 
   53| def conv_receptive_field(
   54|     kernels: Sequence[int], dilations: Sequence[int]
   55| ) -> int:
   74|     if len(kernels) != len(dilations):
   75|         raise ValueError(
   76|             f"kernels and dilations must have equal length, got {len(kernels)} and "
   77|             f"{len(dilations)}; they are positional against each other"
   78|         )
   79|     return 1 + sum((int(kernel) - 1) * int(dilation) for kernel, dilation in zip(kernels, dilations))
   80| 
   82| class CausalConvTransformerEncoder(nn.Module):
  105| 
  106|     def __init__(
  107|         self,
  108|         *,
  109|         d_model: int,
  110|         sequence_length: int,
  111|         conv_kernels: Sequence[int],
  112|         conv_dilations: Sequence[int],
  113|         num_attention_blocks: int,
  114|         num_heads: int,
  115|         d_ff: int,
  116|         attention_window: Optional[int] = None,
  117|         dropout: float = 0.0,
  118|         layer_scale_init: float = LAYER_SCALE_INIT,
  119|         rope_base: float = ROPE_BASE,
  120|     ) -> None:
  141|         super().__init__()
  142|         self.d_model = int(d_model)
  143|         self.sequence_length = int(sequence_length)
  144|         self.conv_kernels = tuple(int(kernel) for kernel in conv_kernels)
  145|         self.conv_dilations = tuple(int(dilation) for dilation in conv_dilations)
  146|         self.num_heads = int(num_heads)
  147|         self.d_ff = int(d_ff)
  148|         self.attention_window = None if attention_window is None else int(attention_window)
  149| 
  152|         self.conv_reach = conv_receptive_field(self.conv_kernels, self.conv_dilations)
  153| 
  154|         if int(num_attention_blocks) < 1:
  155|             raise ValueError(
  156|                 f"num_attention_blocks must be at least 1, got {num_attention_blocks}; an encoder "
  157|                 f"with no attention is the convolution stack this architecture replaces"
  158|             )
  159|         self.num_attention_blocks = int(num_attention_blocks)
  160| 
  161|         self.conv_blocks = nn.ModuleList(
  162|             [
  163|                 GatedCausalConvBlock(
  164|                     d_model,
  165|                     kernel_size=kernel,
  166|                     dilation=dilation,
  167|                     dropout=dropout,
  168|                     layer_scale_init=layer_scale_init,
  169|                 )
  170|                 for kernel, dilation in zip(self.conv_kernels, self.conv_dilations)
  171|             ]
  172|         )
  174|         self.attention_blocks = nn.ModuleList(
  175|             [
  176|                 CausalTransformerBlock(
  177|                     d_model,
  178|                     num_heads,
  179|                     d_ff,
  180|                     self.sequence_length,
  181|                     window=self.attention_window,
  182|                     dropout=dropout,
  183|                     layer_scale_init=layer_scale_init,
  184|                     rope_base=rope_base,
  185|                 )
  186|                 for _ in range(self.num_attention_blocks)
  187|             ]
  188|         )
  189|         self.output_norm = RMSNorm(d_model)
  190| 
  191|     @property
  192|     def receptive_field(self) -> Optional[int]:
  201|         if self.attention_window is None:
  202|             return None
  203|         reach = self.conv_reach + self.num_attention_blocks * (self.attention_window - 1)
  204|         return min(reach, self.sequence_length)
  205| 
  206|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  215|         for block in self.conv_blocks:
  216|             x = block(x)
  217|         for block in self.attention_blocks:
  218|             x = block(x)
  219|         return self.output_norm(x)
  220| 
  221|     def extra_repr(self) -> str:
  223|         context = (
  224|             "full causal prefix"
  225|             if self.attention_window is None
  226|             else f"causal window {self.attention_window}"
  227|         )
  228|         bound = "unbounded" if self.receptive_field is None else f"{self.receptive_field} steps"
  229|         return (
  230|             f"d_model={self.d_model}, conv_blocks={len(self.conv_blocks)}, "
  231|             f"attention_blocks={self.num_attention_blocks}, context={context}, "
  232|             f"receptive_field={bound}"
  233|         )
  234| 
  236| class GatedCausalConvStem(nn.Module):
  259| 
  260|     def __init__(
  261|         self,
  262|         *,
  263|         d_model: int,
  264|         conv_kernels: Sequence[int],
  265|         conv_dilations: Sequence[int],
  266|         dropout: float = 0.0,
  267|         layer_scale_init: float = LAYER_SCALE_INIT,
  268|     ) -> None:
  283|         super().__init__()
  284|         self.d_model = int(d_model)
  285|         self.conv_kernels = tuple(int(kernel) for kernel in conv_kernels)
  286|         self.conv_dilations = tuple(int(dilation) for dilation in conv_dilations)
  288|         self.conv_reach = conv_receptive_field(self.conv_kernels, self.conv_dilations)
  289|         if not self.conv_kernels:
  290|             raise ValueError(
  291|                 "need at least one stem block; a stem of zero blocks is the identity, and the "
  292|                 "identity representation is chosen by name rather than by an empty schedule"
  293|             )
  294| 
  295|         self.conv_blocks = nn.ModuleList(
  296|             [
  297|                 GatedCausalConvBlock(
  298|                     d_model,
  299|                     kernel_size=kernel,
  300|                     dilation=dilation,
  301|                     dropout=dropout,
  302|                     layer_scale_init=layer_scale_init,
  303|                 )
  304|                 for kernel, dilation in zip(self.conv_kernels, self.conv_dilations)
  305|             ]
  306|         )
  307|         self.output_norm = RMSNorm(d_model)
  308| 
  309|     @property
  310|     def receptive_field(self) -> int:
  318|         return self.conv_reach
  319| 
  320|     def forward(self, x: torch.Tensor) -> torch.Tensor:
  329|         for block in self.conv_blocks:
  330|             x = block(x)
  331|         return self.output_norm(x)
  332| 
  333|     def extra_repr(self) -> str:
  335|         return (
  336|             f"d_model={self.d_model}, conv_blocks={len(self.conv_blocks)}, "
  337|             f"receptive_field={self.receptive_field} steps"
  338|         )
  339| 
  341| def _describe(steps: Optional[int]) -> str:
  343|     if steps is None:
  344|         return "unbounded"
  345|     return f"{steps} steps / {steps * SECONDS_PER_STEP:.0f} s"
  346| 
  348| def main() -> None:
  354|     sequence_length = 300
  355|     kernels, dilations = (5, 9), (1, 2)
  356|     target_blocks, source_blocks, source_window = 4, 3, 16
  358|     lag_search_steps = 90
  359| 
  360|     stem = conv_receptive_field(kernels, dilations)
  361|     source_reach = min(stem + source_blocks * (source_window - 1), sequence_length)
  362| 
  363|     print("Causal conv-Transformer encoder receptive fields")
  364|     print(f"  seconds per decimated step : {SECONDS_PER_STEP:.0f}")
  365|     print(f"  sequence length            : {sequence_length} steps")
  366|     print(f"  lag search range           : {_describe(lag_search_steps)}")
  367|     print()
  368|     print(f"  stem, kernels {kernels} dilations {dilations} : {_describe(stem)}")
  369|     print(
  370|         f"  target : {target_blocks} blocks, full causal prefix   -> {_describe(None)}"
  371|     )
  372|     print(
  373|         f"  source : {source_blocks} blocks, causal window {source_window:>2} -> "
  374|         f"{_describe(source_reach)}"
  375|     )
  376|     print()
  377|     print(
  378|         "  The source bound is shorter than the lag search range on purpose: the encoder "
  379|         "characterises\n  a local source neighbourhood and the lag attention selects which "
  380|         "neighbourhood matters."
  381|     )
  382| 
  384| if __name__ == "__main__":
  385|     main()
```

### 10.3 `model.py` — `SeqVaeLagAttnTrfRws`

Same skeleton as §9 with the encoders swapped, `n_depthwise_init`, no `causal_norm`/`decoder_out_channels` keywords, and `_default_decoder_out_channels = raw_per_step`.

`teb_vae/lag_attn_transformer_rws/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   55| from __future__ import annotations
   56| 
   57| import math
   58| from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
   59| 
   60| import torch
   61| from torch import nn
   62| 
   63| from teb_vae.lag_attn.nets.attention import LagCrossAttention
   64| from teb_vae.lag_attn.nets.blocks import initialization, validate_choice
   65| from teb_vae.lag_attn.nets.encoders import LAG_KV_SOURCE_CHOICES
   66| from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
   67| from teb_vae.lag_attn.nets.heads import POSTERIOR_LOGVAR_MODES, PosteriorHead, TEAnalysisHead
   68| from teb_vae.lag_attn.nets.delays import ChannelGate
   69| from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
   70| from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
   71| from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_raw_objective
   72| from teb_vae.lag_attn_rws.nets.losses import horizon_decay_weight
   73| from teb_vae.lag_attn_rws.nets.losses import kld_tensor as closed_form_kld
   74| from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
   75| from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index, build_future_target
   76| from teb_vae.lag_attn_transformer_rws.nets.blocks import init_depthwise_
   77| from teb_vae.lag_attn_transformer_rws.nets.encoders import (
   78|     AvailabilityInputAdapter,
   79|     CausalConvTransformerEncoder,
   80|     GatedCausalConvStem,
   81| )
   82| 
   87| BASE_DECODE_CHOICES = ("sample", "mean")
   88| 
   90| class SeqVaeLagAttnTrfRws(nn.Module):
  102| 
  105|     target_gate: Optional[ChannelGate]
  106|     source_gate: Optional[ChannelGate]
  107| 
  110|     future_index: torch.Tensor
  111| 
  119|     horizon_weight: Optional[torch.Tensor]
  120| 
  121|     def __init__(
  122|         self,
  123|         *,
  124|         sequence_length: int = 300,
  125|         d_model: int = 128,
  126|         d_z: int = 48,
  127|         horizon: int = 30,
  128|         raw_per_step: int = 16,
  129|         warmup_period: int = 30,
  130|         c_y: int = 109,
  131|         c_u: int = 58,
  132|         use_up_st: bool = True,
  133|         max_lag: int = 90,
  134|         num_heads: int = 4,
  135|         d_head: int = 32,
  136|         dropout: float = 0.1,
  137|         decoder_hidden: int = 128,
  138|         horizon_depth: int = 2,
  139|         horizon_kernel: int = 3,
  140|         horizon_film: bool = False,
  141|         horizon_attention_blocks: int = 0,
  142|         horizon_embed_std: float = 0.02,
  143|         head_init_calibration: bool = False,
  144|         a_head_gain: float = 1.0,
  145|         encoder_conv_kernels: Sequence[int] = (5, 9),
  146|         encoder_conv_dilations: Sequence[int] = (1, 2),
  147|         encoder_num_heads: int = 4,
  148|         encoder_d_ff: int = 256,
  149|         target_attention_blocks: int = 4,
  150|         source_attention_blocks: int = 3,
  151|         source_attention_window: Optional[int] = 16,
  152|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  153|         mu_scale: float = 5.0,
  154|         delta_mu_scale: float = 3.0,
  155|         delta_logvar_scale: float = 2.0,
  156|         posterior_logvar_mode: str = "residual",
  157|         source_dropout: Optional[float] = None,
  158|         lag_kv_source: str = "encoder",
  159|         use_entmax: bool = False,
  160|         attention_grad_checkpoint: bool = False,
  161|         lag_bias_init: str = "normal",
  162|         alibi_slope_scale: float = 1.0,
  163|         query_uses_logvar: bool = False,
  164|         prior_availability_input: bool = False,
  165|         coverage_floor: float = 0.9,
  166|         base_decode: str = "sample",
  167|         persistence_residual: bool = False,
  168|         horizon_weight_halflife_steps: Optional[float] = None,
  169|         target_keep_index: Optional[Sequence[int]] = None,
  170|         target_delays: Optional[Sequence[int]] = None,
  171|         source_keep_index: Optional[Sequence[int]] = None,
  172|         source_delays: Optional[Sequence[int]] = None,
  173|         init_weights: bool = True,
  174|     ) -> None:
  343|         super().__init__()
  344| 
  350|         if int(c_y) < 1 or int(c_u) < 1:
  351|             raise ValueError(
  352|                 f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
  353|             )
  354|         if int(num_heads) * int(d_head) != int(d_model):
  355|             raise ValueError(
  356|                 f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
  357|             )
  361|         if int(max_lag) < 0:
  362|             raise ValueError(f"max_lag must be >= 0, got {max_lag}")
  365|         if int(d_z) % int(num_heads) != 0:
  366|             raise ValueError(
  367|                 f"the head-structured latent requires d_z % num_heads == 0, "
  368|                 f"got d_z={d_z}, num_heads={num_heads}"
  369|             )
  370| 
  374|         self.geometry = TrimmedRawGeometry(
  375|             raw_len=int(sequence_length) * int(raw_per_step),
  376|             decimation=int(raw_per_step),
  377|             horizon=int(horizon),
  378|             warmup=int(warmup_period),
  379|         )
  380| 
  381|         self.sequence_length = int(sequence_length)
  382|         self.d_model = int(d_model)
  383|         self.d_z = int(d_z)
  384|         self.horizon = int(horizon)
  385|         self.raw_per_step = int(raw_per_step)
  386|         self.warmup_period = int(warmup_period)
  387|         self.c_y = int(c_y)
  388|         self.c_u = int(c_u)
  389|         self.use_up_st = bool(use_up_st)
  390|         self.max_lag = int(max_lag)
  391|         self.num_heads = int(num_heads)
  392|         self.mu_scale = float(mu_scale)
  393|         self.delta_mu_scale = float(delta_mu_scale)
  394|         self.delta_logvar_scale = float(delta_logvar_scale)
  395|         self.posterior_logvar_mode = validate_choice(
  396|             posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
  397|         )
  410|         self.source_dropout = float(dropout if source_dropout is None else source_dropout)
  411|         self.posterior_source_dropout = 0.0 if source_dropout is None else float(source_dropout)
  412|         self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
  413|         self.coverage_floor = float(coverage_floor)
  414|         self.base_decode = validate_choice(base_decode, BASE_DECODE_CHOICES, "base_decode")
  419|         self.persistence_residual = bool(persistence_residual)
  420|         if self.persistence_residual:
  421|             self._check_persistence_target()
  425|         self.horizon_embed_std = float(horizon_embed_std)
  426|         self.head_init_calibration = bool(head_init_calibration)
  427|         self.a_head_gain = float(a_head_gain)
  428| 
  433|         self.register_buffer(
  434|             "future_index", build_future_index(self.geometry), persistent=False
  435|         )
  436| 
  449|         self.horizon_weight_halflife_steps = (
  450|             None
  451|             if horizon_weight_halflife_steps is None
  452|             else float(horizon_weight_halflife_steps)
  453|         )
  454|         if self.horizon_weight_halflife_steps is not None:
  455|             self.register_buffer(
  456|                 "horizon_weight",
  457|                 horizon_decay_weight(self.horizon_weight_halflife_steps, self.horizon),
  458|                 persistent=False,
  459|             )
  460| 
  465|         self.target_gate = self._build_channel_gate(
  466|             self.c_y, target_keep_index, target_delays
  467|         )
  468|         self.source_gate = self._build_channel_gate(
  469|             self.c_u, source_keep_index, source_delays
  470|         )
  471| 
  476|         self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)
  477|         self.source_adapter = self._build_adapter(
  478|             self.source_gate, self.c_u, self.source_dropout
  479|         )
  480| 
  484|         self.target_encoder = CausalConvTransformerEncoder(
  485|             d_model=d_model,
  486|             sequence_length=self.sequence_length,
  487|             conv_kernels=encoder_conv_kernels,
  488|             conv_dilations=encoder_conv_dilations,
  489|             num_attention_blocks=target_attention_blocks,
  490|             num_heads=encoder_num_heads,
  491|             d_ff=encoder_d_ff,
  492|             attention_window=None,
  493|             dropout=dropout,
  494|         )
  500|         self.lag_kv_source = validate_choice(
  501|             lag_kv_source, LAG_KV_SOURCE_CHOICES, "lag_kv_source"
  502|         )
  503|         if self.lag_kv_source == "encoder":
  504|             self.source_encoder = CausalConvTransformerEncoder(
  505|                 d_model=d_model,
  506|                 sequence_length=self.sequence_length,
  507|                 conv_kernels=encoder_conv_kernels,
  508|                 conv_dilations=encoder_conv_dilations,
  509|                 num_attention_blocks=source_attention_blocks,
  510|                 num_heads=encoder_num_heads,
  511|                 d_ff=encoder_d_ff,
  512|                 attention_window=source_attention_window,
  513|                 dropout=self.source_dropout,
  514|             )
  515|         elif self.lag_kv_source == "conv_stem":
  518|             self.source_kv_stem = GatedCausalConvStem(
  519|                 d_model=d_model,
  520|                 conv_kernels=encoder_conv_kernels,
  521|                 conv_dilations=encoder_conv_dilations,
  522|                 dropout=self.source_dropout,
  523|             )
  524| 
  528|         self.prior_availability_input = bool(prior_availability_input)
  529|         self.prior_head = FullLatentPriorHead(
  530|             d_model=d_model,
  531|             d_z=d_z,
  532|             logvar_clamp=logvar_clamp,
  533|             dropout=dropout,
  534|             mu_scale=self.mu_scale,
  535|             clock_dim=self._prior_clock_dim() if self.prior_availability_input else None,
  536|         )
  537| 
  542|         self.query_uses_logvar = bool(query_uses_logvar)
  543|         query_in = 2 * self.d_z if self.query_uses_logvar else self.d_z
  544|         self.query_proj = nn.Linear(query_in, d_model)
  545| 
  549|         self.lag_attn = LagCrossAttention(
  550|             d_model=d_model,
  551|             num_heads=num_heads,
  552|             d_head=d_head,
  553|             max_lag=max_lag,
  554|             dropout=0.0,
  555|             use_entmax=use_entmax,
  556|             grad_checkpoint=attention_grad_checkpoint,
  557|             lag_bias_init=lag_bias_init,
  558|             alibi_slope_scale=alibi_slope_scale,
  559|         )
  560|         self.posterior_head = PosteriorHead(
  561|             d_model=d_model,
  562|             d_z=d_z,
  563|             logvar_clamp=logvar_clamp,
  564|             dropout=dropout,
  565|             delta_mu_scale=self.delta_mu_scale,
  566|             head_structured=True,
  567|             num_heads=num_heads,
  568|             d_head=d_head,
  569|             delta_logvar_scale=self.delta_logvar_scale,
  570|             posterior_logvar_mode=self.posterior_logvar_mode,
  573|             source_dropout=self.posterior_source_dropout,
  574|         )
  575|         self.te_analysis = TEAnalysisHead()
  576| 
  586|         self.horizon_core = HorizonDecoderCore(
  587|             d_hidden=decoder_hidden,
  588|             horizon=horizon,
  589|             kernel_size=horizon_kernel,
  590|             depth=horizon_depth,
  591|             film=horizon_film,
  592|             film_per_block=True,
  593|             attention_blocks=horizon_attention_blocks,
  594|         )
  595|         self.decoder_out_channels = self._default_decoder_out_channels()
  596|         self.decoder = BaselineFutureDecoder(
  597|             core=self.horizon_core,
  598|             d_model=d_z,
  599|             out_channels=self.decoder_out_channels,
  600|             d_hidden=decoder_hidden,
  601|             dropout=0.0,
  602|             logvar_clamp=logvar_clamp,
  605|             persistence_residual=self.persistence_residual,
  606|         )
  607| 
  611|         for parameter in self.lag_attn.W_o.parameters():
  612|             parameter.requires_grad_(False)
  613| 
  619|         self.n_depthwise_init = 0
  620|         if init_weights:
  621|             initialization(self)
  629|             self.n_depthwise_init = init_depthwise_(self)
  632|         self._zero_init_delta_heads()
  637|         self.prior_head.zero_init_clock()
  643|         self._zero_init_film_generators()
  644| 
  649|         if self.horizon_embed_std != 0.02:
  650|             self._reinit_horizon_embedding()
  651|         if self.head_init_calibration:
  652|             self._calibrate_output_heads()
  653|             self._calibrate_prior_scale()
  654|         if self.a_head_gain != 1.0:
  655|             self._set_a_head_gain()
  656| 
  657|     def _default_decoder_out_channels(self) -> int:
  681|         return self.raw_per_step
  682| 
  683|     @staticmethod
  684|     def _build_channel_gate(
  685|         declared_width: int,
  686|         keep_index: Optional[Sequence[int]],
  687|         delays: Optional[Sequence[int]],
  688|     ) -> Optional[ChannelGate]:
  708|         if keep_index is None and delays is None:
  709|             return None
  710|         return ChannelGate(
  711|             declared_width=int(declared_width), keep_index=keep_index, delays=delays
  712|         )
  713| 
  714|     def _build_adapter(
  715|         self, gate: Optional[ChannelGate], declared_width: int, dropout: float
  716|     ) -> AvailabilityInputAdapter:
  733|         width = declared_width if gate is None else gate.out_channels
  734|         delays = None if gate is None else [int(value) for value in gate.delay.delay_steps]
  735|         return AvailabilityInputAdapter(
  736|             in_dim=width,
  737|             d_model=self.d_model,
  738|             sequence_length=self.sequence_length,
  739|             dropout=dropout,
  740|             delays=delays,
  741|         )
  742| 
  743|     def source_kv_body(self) -> Optional[nn.Module]:
  755|         if self.lag_kv_source == "encoder":
  756|             return self.source_encoder
  757|         if self.lag_kv_source == "conv_stem":
  758|             return self.source_kv_stem
  759|         return None
  760| 
  761|     def source_kv_modules(self) -> Tuple[nn.Module, ...]:
  772|         body = self.source_kv_body()
  773|         return (self.source_adapter,) if body is None else (self.source_adapter, body)
  774| 
  775|     def encode_source_kv(self, source: torch.Tensor) -> torch.Tensor:
  786|         encoded = source
  787|         for module in self.source_kv_modules():
  788|             encoded = module(encoded)
  789|         return encoded
  790| 
  791|     def build_lag_mask(
  792|         self, seq_len: int, device: Optional[torch.device] = None
  793|     ) -> torch.Tensor:
  810|         return self.lag_attn.build_lag_mask(seq_len, device=device)
  811| 
  812|     @property
  813|     def source_delay_steps(self) -> int:
  838|         return 0 if self.source_gate is None else self.source_gate.max_delay
  839| 
  840|     @staticmethod
  841|     def _zero_linear(layer: nn.Linear) -> None:
  843|         nn.init.zeros_(layer.weight)
  844|         if layer.bias is not None:
  845|             nn.init.zeros_(layer.bias)
  846| 
  847|     def _prior_clock_dim(self) -> int:
  869|         raise ValueError(
  870|             f"prior_availability_input=True on {type(self).__name__}, whose input streams carry no "
  871|             f"warm-up: every channel is honest at every step, so nothing arrives late and there is "
  872|             f"no availability term in the KL to cancel. The flag belongs to the causal cells, whose "
  873|             f"one-sided inputs arrive over the first steps of a segment."
  874|         )
  875| 
  876|     def _check_persistence_target(self) -> None:
  898|         raise ValueError(
  899|             f"persistence_residual=True on {type(self).__name__}, whose forecast block is R raw "
  900|             f"samples of one signal per horizon token: there is no per-channel level for a "
  901|             f"persistence term to carry forward. The flag belongs to the feature-target cells, "
  902|             f"whose block's last axis counts stored coefficients of the target itself."
  903|         )
  904| 
  905|     def _zero_init_delta_heads(self) -> None:
  916|         for module in (self.posterior_head.delta_mu_head, self.posterior_head.delta_logvar_head):
  919|             if module is None:
  920|                 continue
  921|             layers = list(module) if isinstance(module, nn.ModuleList) else [module]
  922|             for layer in layers:
  923|                 self._zero_linear(cast(nn.Linear, layer))
  924| 
  932|         independent = self.posterior_head.logvar_post_head
  933|         if independent is not None:
  936|             lo, hi = self.posterior_head.logvar_clamp
  937|             if not lo < 0.0 < hi:
  938|                 raise ValueError(
  939|                     f"posterior_logvar_mode='independent' seeds the head at unit scale, which "
  940|                     f"needs 0 inside logvar_clamp; got ({lo}, {hi})"
  941|                 )
  942|             bias_value = math.log((0.0 - lo) / (hi - 0.0))
  943|             layers = (
  944|                 list(independent)
  945|                 if isinstance(independent, nn.ModuleList)
  946|                 else [independent]
  947|             )
  948|             for layer in layers:
  949|                 linear = cast(nn.Linear, layer)
  950|                 nn.init.zeros_(linear.weight)
  951|                 linear.bias.data.fill_(bias_value)
  952| 
  953|     def _zero_init_film_generators(self) -> None:
  964|         core = self.horizon_core
  965|         film_layers: List[nn.Module] = []
  966|         if core.film_gen is not None:
  967|             film_layers.append(core.film_gen)
  968|         if core.refine.film is not None:
  969|             film_layers.extend(core.refine.film)
  970|         for layer in film_layers:
  971|             self._zero_linear(cast(nn.Linear, layer))
  972| 
  973|     def _reinit_horizon_embedding(self) -> None:
  984|         nn.init.normal_(
  985|             self.horizon_core.horizon_embedding, mean=0.0, std=self.horizon_embed_std
  986|         )
  987| 
  988|     def _calibrate_output_heads(self) -> None:
 1004|         self.decoder.mean_head.weight.data.mul_(0.02)
 1005|         self.decoder.logvar_head.bias.data.fill_(math.log(5.0 / 3.0))
 1006|         self.decoder.logvar_head.weight.data.mul_(0.1)
 1007| 
 1008|     def _calibrate_prior_scale(self) -> None:
 1034|         lo, hi = self.logvar_clamp
 1035|         if not lo < 0.0 < hi:
 1036|             raise ValueError(
 1037|                 f"prior scale calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
 1038|             )
 1039|         head = self.prior_head.logvar_prior_head
 1040|         if not isinstance(head.skip_proj, nn.Linear):
 1041|             raise ValueError(
 1042|                 "prior scale calibration requires a projected skip on the log-variance head; "
 1043|                 "with d_model == d_z the skip is an identity and the output cannot be pinned"
 1044|             )
 1045|         self._zero_linear(head.skip_proj)
 1046|         final = cast(nn.Linear, head.body[-1])
 1047|         nn.init.zeros_(final.weight)
 1048|         final.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))
 1049| 
 1050|     def _set_a_head_gain(self) -> None:
 1062|         nn.init.constant_(self.posterior_head.a_head_norm.weight, self.a_head_gain)
 1063| 
 1064|     def _reparameterize_shared(
 1065|         self,
 1066|         mu_prior: torch.Tensor,
 1067|         logvar_prior: torch.Tensor,
 1068|         mu_post: torch.Tensor,
 1069|         logvar_post: torch.Tensor,
 1070|     ) -> Tuple[torch.Tensor, torch.Tensor]:
 1098|         epsilon = torch.randn_like(mu_prior)
 1099|         z_post = mu_post + epsilon * torch.exp(0.5 * logvar_post)
 1100|         if self.base_decode == "mean":
 1104|             return mu_prior, z_post
 1105|         z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
 1106|         return z_prior, z_post
 1107| 
 1108|     def forward(
 1109|         self,
 1110|         y_st: torch.Tensor,
 1111|         y_ph: torch.Tensor,
 1112|         u_stream: torch.Tensor,
 1113|     ) -> Dict[str, torch.Tensor]:
 1159|         target = torch.cat([y_st, y_ph], dim=-1)
 1160|         if self.target_gate is not None:
 1161|             target = self.target_gate(target)
 1162|         source = u_stream if self.source_gate is None else self.source_gate(u_stream)
 1163| 
 1164|         h_y = self.target_encoder(self.target_adapter(target))
 1165|         h_u = self.encode_source_kv(source)
 1166| 
 1167|         mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(h_y)
 1168| 
 1172|         query = (
 1173|             torch.cat([mu_prior, logvar_prior], dim=-1)
 1174|             if self.query_uses_logvar
 1175|             else mu_prior
 1176|         )
 1177|         _, alpha, attended_heads = self.lag_attn(
 1178|             self.query_proj(query), h_u, self.build_lag_mask(h_u.shape[1], h_u.device)
 1179|         )
 1180| 
 1181|         mu_post, logvar_post = self.posterior_head(
 1182|             h_y, attended_heads, mu_prior, raw_logvar_prior
 1183|         )
 1184|         z_prior, z_post = self._reparameterize_shared(
 1185|             mu_prior, logvar_prior, mu_post, logvar_post
 1186|         )
 1187| 
 1190|         with torch.no_grad():
 1191|             mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
 1192|             delta_mu_sat_frac = (
 1193|                 (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
 1194|             ).float().mean()
 1195| 
 1198|         t_valid = self.geometry.t_valid
 1199|         mu_base, logvar_base = self.decoder(z_prior[:, :t_valid])
 1200|         mu_full, logvar_full = self.decoder(z_post[:, :t_valid])
 1201| 
 1205|         kld_btd = self.kld_tensor(
 1206|             mu_prior=mu_prior,
 1207|             logvar_prior=logvar_prior,
 1208|             mu_post=mu_post,
 1209|             logvar_post=logvar_post,
 1210|         )
 1211|         kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
 1212|             kld_btd, alpha, head_structured=True
 1213|         )
 1214| 
 1215|         return {
 1216|             "mu_prior": mu_prior,
 1217|             "logvar_prior": logvar_prior,
 1218|             "raw_logvar_prior": raw_logvar_prior,
 1219|             "mu_post": mu_post,
 1220|             "logvar_post": logvar_post,
 1221|             "z_prior": z_prior,
 1222|             "z_post": z_post,
 1223|             "target_state": h_y,
 1224|             "source_state": h_u,
 1225|             "attended_source_heads": attended_heads,
 1226|             "attn_weights": alpha,
 1227|             "mu_base": mu_base,
 1228|             "logvar_base": logvar_base,
 1229|             "mu_full": mu_full,
 1230|             "logvar_full": logvar_full,
 1231|             "kld_per_t": kld_per_t,
 1232|             "kld_per_t_per_head": kld_per_t_per_head,
 1233|             "source_kl_lag_map": source_kl_lag_map,
 1234|             "mu_prior_sat_frac": mu_prior_sat_frac,
 1235|             "delta_mu_sat_frac": delta_mu_sat_frac,
 1236|         }
 1237| 
 1238|     def kld_tensor(
 1239|         self,
 1240|         mu_prior: torch.Tensor,
 1241|         logvar_prior: torch.Tensor,
 1242|         mu_post: torch.Tensor,
 1243|         logvar_post: torch.Tensor,
 1244|     ) -> torch.Tensor:
 1261|         return closed_form_kld(mu_prior, logvar_prior, mu_post, logvar_post)
 1262| 
 1263|     def compute_loss(
 1264|         self,
 1265|         forward_outputs: Dict[str, torch.Tensor],
 1266|         fhr_raw: torch.Tensor,
 1267|         *,
 1268|         weight: torch.Tensor,
 1269|         beta: float = 1.0,
 1270|         beta_prior: float = 0.0,
 1271|         lambda_full: float = 1.0,
 1272|         lambda_base: float = 1.0,
 1273|         likelihood: str = "gaussian_nll",
 1274|         free_bits: float = 0.0,
 1275|         lambda_ms: float = 0.0,
 1276|         lambda_deriv: float = 0.0,
 1277|         lambda_boundary: float = 0.0,
 1278|     ) -> Dict[str, Any]:
 1317|         return compute_raw_objective(
 1318|             forward_outputs,
 1319|             build_future_target(fhr_raw, self.geometry, future_index=self.future_index),
 1320|             weight=weight,
 1321|             geometry=self.geometry,
 1323|             block_width=self.geometry.r,
 1324|             coverage_floor=self.coverage_floor,
 1325|             logvar_clamp=self.logvar_clamp,
 1326|             beta=beta,
 1327|             beta_prior=beta_prior,
 1328|             lambda_full=lambda_full,
 1329|             lambda_base=lambda_base,
 1330|             likelihood=likelihood,
 1331|             free_bits=free_bits,
 1332|             lambda_ms=lambda_ms,
 1333|             lambda_deriv=lambda_deriv,
 1334|             lambda_boundary=lambda_boundary,
 1339|             horizon_weight=getattr(self, "horizon_weight", None),
 1340|         )
```

## 11. Feature-target mixin — `teb_vae/lag_attn_fs/nets/feature_target.py` (`FeatureForecastTarget`)

Decoder width hook (kept target channels), the stored-clock target gather $Y^+[b,a,\tau,k]=Y[b,t_a+1+\tau,\mathrm{keep}[k]]$, `compute_loss` delegation with `block_width=C_keep` and the channel/horizon weights.

`teb_vae/lag_attn_fs/nets/feature_target.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   51| from __future__ import annotations
   52| 
   53| from typing import Any, Dict, Optional, Tuple
   54| 
   55| import torch
   56| 
   57| from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective
   58| from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
   59| from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask
   60| 
   62| class FeatureForecastTarget:
   72| 
   88|     TARGET_BLOCK_SPLIT: int = 43
   89| 
  103|     target_channel_weight: Optional[torch.Tensor]
  104| 
  105|     def _default_decoder_out_channels(self) -> int:
  119|         return self.c_y if self.target_gate is None else self.target_gate.out_channels
  120| 
  121|     def _build_forecast_target(
  122|         self, target_features: torch.Tensor, anchors: Optional[torch.Tensor] = None
  123|     ) -> torch.Tensor:
  159|         if target_features.dim() != 3:
  160|             raise ValueError(
  161|                 f"target stream must be 3-D (B, T, c_y), got shape {tuple(target_features.shape)}"
  162|             )
  163|         if target_features.size(1) != self.geometry.t:
  164|             raise ValueError(
  165|                 f"target stream length {target_features.size(1)} != geometry.t "
  166|                 f"{self.geometry.t}; this geometry assumes the trimmed loader "
  167|                 f"(trim_minutes: 1.0 -> T = {self.geometry.t} decimated steps), so a mismatch "
  168|                 f"means the loader ran at a different trim_minutes"
  169|             )
  170|         if target_features.size(2) != self.c_y:
  171|             raise ValueError(
  172|                 f"target stream has {target_features.size(2)} channels but the model declares "
  173|                 f"c_y={self.c_y}; the surviving-channel index is positional into the declared "
  174|                 f"width, so a mismatch would gather the wrong channels rather than fail"
  175|             )
  176| 
  177|         gathered = (
  178|             target_features
  179|             if self.target_gate is None
  180|             else torch.index_select(target_features, -1, self.target_gate.keep_index)
  181|         )
  182|         if anchors is None:
  185|             return (
  186|                 gathered[:, 1:, :]
  187|                 .unfold(dimension=1, size=self.horizon, step=1)
  188|                 .permute(0, 1, 3, 2)
  189|             )
  190| 
  191|         batch, channels = gathered.shape[0], gathered.shape[-1]
  192|         steps = torch.arange(self.horizon, device=gathered.device)
  194|         time_index = anchors.to(torch.long)[:, :, None] + 1 + steps[None, None, :]
  195|         window = gathered.gather(
  196|             1, time_index.reshape(batch, -1, 1).expand(-1, -1, channels)
  197|         )
  198|         return window.reshape(batch, anchors.shape[1], self.horizon, channels)
  199| 
  256| 
  325| 
  326|     def compute_loss(
  327|         self,
  328|         forward_outputs: Dict[str, torch.Tensor],
  329|         target_features: torch.Tensor,
  330|         *,
  331|         weight: torch.Tensor,
  332|         beta: float = 1.0,
  333|         beta_prior: float = 0.0,
  334|         lambda_full: float = 1.0,
  335|         lambda_base: float = 1.0,
  336|         likelihood: str = "gaussian_nll",
  337|         free_bits: float = 0.0,
  338|         lambda_ms: float = 0.0,
  339|         lambda_deriv: float = 0.0,
  340|         lambda_boundary: float = 0.0,
  341|     ) -> Dict[str, Any]:
  400|         target = self._build_forecast_target(
  401|             target_features, forward_outputs.get("anchor_index")
  402|         )
  403|         result = compute_shared_objective(
  404|             forward_outputs,
  405|             target,
  406|             weight=weight,
  407|             geometry=self.geometry,
  410|             block_width=self.decoder_out_channels,
  411|             coverage_floor=self.coverage_floor,
  412|             logvar_clamp=self.logvar_clamp,
  413|             beta=beta,
  414|             beta_prior=beta_prior,
  415|             lambda_full=lambda_full,
  416|             lambda_base=lambda_base,
  417|             likelihood=likelihood,
  418|             free_bits=free_bits,
  419|             lambda_ms=lambda_ms,
  420|             lambda_deriv=lambda_deriv,
  421|             lambda_boundary=lambda_boundary,
  425|             channel_weight=getattr(self, "target_channel_weight", None),
  428|             horizon_weight=getattr(self, "horizon_weight", None),
  429|         )
  433|         result["metrics"].update(
  434|             self._resolved_forecast_gaps(
  435|                 forward_outputs, target, weight, likelihood=likelihood
  436|             )
  437|         )
  438|         return result
```

## 12. Causal input mixin — `teb_vae/lag_attn_cfs/nets/causal_inputs.py` (`CausalWarmupInputs`)

Owns: `_set_causal_inputs`, `anchor_ceiling`, `_validate_causal_geometry`, `_combined_source_steps` ($W'_c+d_c$), `_prior_clock` (zeroed-source encode), `_prior_clock_dim`, `_build_adapter` override (availability at $W'_c+d_c$), `build_lag_mask` (floor), `_build_anchor_index` (tiling), and the **tiled `forward` that all four models run**.

`teb_vae/lag_attn_cfs/nets/causal_inputs.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   70| from __future__ import annotations
   71| 
   72| from typing import Dict, Optional, Sequence, Tuple, Union
   73| 
   74| import torch
   75| 
   76| from teb_vae.lag_attn.nets.delays import ChannelGate
   77| from teb_vae.lag_attn.nets.encoders import AvailabilityInputAdapter
   78| from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
   79| 
   92| CAUSAL_ONLY_KEYWORDS: Tuple[str, ...] = (
   93|     "target_warmup_steps",
   94|     "source_warmup_steps",
   95|     "anchor_stride",
   96|     "lag_floor",
   97|     "target_weight_st",
   98|     "target_weight_ph",
   99|     "target_align_delays",
  100|     "source_align_delays",
  101|     "target_novelty_frac",
  102|     "target_forecast_shift",
  103| )
  104| 
  108| FORWARDED_EXCLUSIONS: Tuple[str, ...] = ("self", "__class__") + CAUSAL_ONLY_KEYWORDS
  109| 
  111| class CausalWarmupInputs:
  137| 
  138|     target_warmup_steps: Optional[Tuple[int, ...]]
  139|     source_warmup_steps: Optional[Tuple[int, ...]]
  140|     anchor_stride: int
  141|     lag_floor: int
  142|     target_forecast_shift: Optional[Tuple[int, ...]]
  143| 
  147|     def _set_causal_inputs(
  148|         self,
  149|         *,
  150|         horizon: int,
  151|         target_keep_index: Optional[Sequence[int]],
  152|         target_warmup_steps: Optional[Sequence[int]],
  153|         source_keep_index: Optional[Sequence[int]],
  154|         source_warmup_steps: Optional[Sequence[int]],
  155|         anchor_stride: int,
  156|         lag_floor: int,
  157|         target_forecast_shift: Optional[Sequence[int]] = None,
  158|     ) -> None:
  190|         if not 1 <= int(anchor_stride) <= int(horizon):
  191|             raise ValueError(
  192|                 f"anchor_stride must be in [1, horizon] = [1, {int(horizon)}], got "
  193|                 f"{anchor_stride}. Below 1 there is no anchor set; above the horizon the decoded "
  194|                 f"windows leave gaps, so target steps between two tiles would never be scored at "
  195|                 f"any phase."
  196|             )
  197|         if int(lag_floor) < 0:
  198|             raise ValueError(f"lag_floor must be >= 0, got {lag_floor}")
  199| 
  203|         for stream, warmup, keep in (
  204|             ("target", target_warmup_steps, target_keep_index),
  205|             ("source", source_warmup_steps, source_keep_index),
  206|         ):
  207|             if warmup is not None and keep is None:
  208|                 raise ValueError(
  209|                     f"{stream}_warmup_steps was given without {stream}_keep_index. The warm-up "
  210|                     f"vector is positional against the surviving channels, so the two are "
  211|                     f"resolved together and travel together; a keep-index covering every channel "
  212|                     f"is still an explicit keep-index."
  213|                 )
  214| 
  221|         if target_forecast_shift is not None:
  222|             if target_keep_index is None:
  223|                 raise ValueError(
  224|                     "target_forecast_shift was given without target_keep_index. The shift is "
  225|                     "positional against the surviving target channels, so the two are resolved "
  226|                     "together and travel together."
  227|                 )
  228|             shifts = tuple(int(step) for step in target_forecast_shift)
  229|             if shifts and min(shifts) < 0 < max(shifts):
  230|                 raise ValueError(
  231|                     f"target_forecast_shift mixes signs (min {min(shifts)}, max {max(shifts)}). "
  232|                     f"Each forecast clock is single-signed -- 'physical' advances, 'input' delays "
  233|                     f"-- so a mixed vector names no clock and its two halves would be checked "
  234|                     f"against each other's geometry."
  235|                 )
  236|             target_forecast_shift = shifts
  237| 
  238|         self.target_warmup_steps = None if target_warmup_steps is None else tuple(
  239|             int(step) for step in target_warmup_steps
  240|         )
  241|         self.source_warmup_steps = None if source_warmup_steps is None else tuple(
  242|             int(step) for step in source_warmup_steps
  243|         )
  244|         self.anchor_stride = int(anchor_stride)
  245|         self.lag_floor = int(lag_floor)
  246|         self.target_forecast_shift = target_forecast_shift
  247| 
  248|     @property
  249|     def anchor_ceiling(self) -> int:
  262|         shift = getattr(self, "target_forecast_shift", None)
  263|         if not shift:
  264|             return int(self.geometry.t_valid)
  265|         return int(self.geometry.t_valid) - max(0, max(shift))
  266| 
  267|     def _validate_causal_geometry(self) -> None:
  290|         if self.anchor_stride > self.anchor_ceiling - self.warmup_period:
  291|             advance = (
  292|                 ""
  293|                 if self.anchor_ceiling == self.geometry.t_valid
  294|                 else (
  295|                     f" (T_valid {self.geometry.t_valid} less the forecast clock's largest "
  296|                     f"advance {self.geometry.t_valid - self.anchor_ceiling})"
  297|                 )
  298|             )
  299|             raise ValueError(
  300|                 f"anchor_stride={self.anchor_stride} leaves no anchor at phase "
  301|                 f"{self.anchor_stride - 1}: the first would be "
  302|                 f"{self.warmup_period + self.anchor_stride - 1}, against an anchor ceiling of "
  303|                 f"{self.anchor_ceiling}{advance} and warmup_period={self.warmup_period}. A "
  304|                 f"sample drawn at that phase would contribute no forecast at all, and nothing "
  305|                 f"downstream reports an empty anchor row."
  306|             )
  307|         floor_args = [
  308|             self.warmup_period,
  309|             self.target_warmup_steps or (),
  310|             ()
  311|             if self.target_gate is None
  312|             else tuple(int(shift) for shift in self.target_gate.delay.delay_steps),
  313|         ]
  319|         if self.target_forecast_shift:
  320|             floor_args.append(self.target_forecast_shift)
  321|         self._check_anchor_floor(*floor_args)
  322|         self._resolve_warmup_readout_constants()
  323| 
  324|     def _combined_source_steps(self) -> Optional[Tuple[int, ...]]:
  343|         waits = self.source_warmup_steps
  344|         if waits is None or self.source_gate is None:
  345|             return waits
  346|         return tuple(
  347|             wait + int(shift)
  348|             for wait, shift in zip(waits, self.source_gate.delay.delay_steps)
  349|         )
  350| 
  351|     def _prior_clock(self, u_stream: torch.Tensor) -> torch.Tensor:
  404|         zeros = u_stream.new_zeros((1, *u_stream.shape[1:]))
  405|         gated = zeros if self.source_gate is None else self.source_gate(zeros)
  410|         modules = self.source_kv_modules()
  411|         was_training = [module.training for module in modules]
  412|         for module in modules:
  413|             module.eval()
  414|         try:
  415|             with torch.no_grad():
  416|                 encoded = self.encode_source_kv(gated)
  417|         finally:
  418|             for module, flag in zip(modules, was_training):
  419|                 module.train(flag)
  420|         return encoded.detach()
  421| 
  422|     def _prior_clock_dim(self) -> int:
  434|         return int(self.d_model)
  435| 
  569| 
  573|     def _build_adapter(
  574|         self, gate: Optional[ChannelGate], declared_width: int, dropout: float
  575|     ) -> AvailabilityInputAdapter:
  611|         warmup = (
  612|             self.target_warmup_steps
  613|             if gate is self.target_gate
  614|             else self.source_warmup_steps
  615|         )
  616|         if warmup is None:
  617|             return super()._build_adapter(gate, declared_width, dropout)
  618| 
  619|         width = declared_width if gate is None else gate.out_channels
  620|         delays = list(warmup)
  621|         if gate is not None:
  622|             delays = [
  623|                 wait + int(shift) for wait, shift in zip(delays, gate.delay.delay_steps)
  624|             ]
  625|         return AvailabilityInputAdapter(
  626|             in_dim=width,
  627|             d_model=self.d_model,
  628|             sequence_length=self.sequence_length,
  629|             dropout=dropout,
  630|             delays=delays,
  631|         )
  632| 
  633|     def build_lag_mask(
  634|         self, seq_len: int, device: Optional[torch.device] = None
  635|     ) -> torch.Tensor:
  654|         mask = super().build_lag_mask(seq_len, device=device)
  655|         if self.lag_floor == 0:
  656|             return mask
  657|         steps = torch.arange(seq_len, device=device)[:, None]
  658|         lags = torch.arange(self.lag_attn.L, device=device)[None, :]
  659|         return mask & (steps - lags >= self.lag_floor)
  660| 
  664|     def _build_anchor_index(
  665|         self,
  666|         batch: int,
  667|         device: torch.device,
  668|         anchor_phase: Optional[Union[int, torch.Tensor]] = None,
  669|         anchor_stride: Optional[int] = None,
  670|     ) -> Tuple[torch.Tensor, torch.Tensor]:
  706|         stride = self.anchor_stride if anchor_stride is None else int(anchor_stride)
  707|         if not 1 <= stride <= self.horizon:
  708|             raise ValueError(
  709|                 f"anchor_stride must be in [1, horizon] = [1, {self.horizon}], got {stride}"
  710|             )
  711| 
  715|         floor, t_valid = self.warmup_period, self.anchor_ceiling
  716|         span = t_valid - floor
  717| 
  718|         if anchor_phase is None:
  722|             if stride > 1:
  723|                 raise ValueError(
  724|                     f"anchor_phase is required at anchor_stride={stride}: without it every sample "
  725|                     f"would be decoded at the same tile grid forever, at a fixed offset from the "
  726|                     f"segment start, and no shape or count would differ. Pass a (B,) phase, or "
  727|                     f"decode densely with anchor_stride=1."
  728|                 )
  729|             phase = torch.zeros(batch, dtype=torch.long, device=device)
  730|         elif isinstance(anchor_phase, torch.Tensor):
  731|             phase = anchor_phase.to(device=device, dtype=torch.long).reshape(-1)
  732|             if phase.numel() != batch:
  733|                 raise ValueError(
  734|                     f"anchor_phase has {phase.numel()} entries but the batch is {batch}; the "
  735|                     f"phase is per sample, so a mismatch would tile one sample at another's grid"
  736|                 )
  737|         else:
  738|             phase = torch.full((batch,), int(anchor_phase), dtype=torch.long, device=device)
  739| 
  740|         if bool(((phase < 0) | (phase >= stride)).any()):
  741|             offending = int(phase[(phase < 0) | (phase >= stride)][0])
  742|             raise ValueError(
  743|                 f"anchor_phase {offending} is outside [0, anchor_stride) = [0, {stride}). The "
  744|                 f"anchor set truncates rather than rotating, so a phase at or above the stride "
  745|                 f"drops leading anchors instead of shifting the grid -- and at stride 1 the only "
  746|                 f"admissible phase is 0."
  747|             )
  748| 
  749|         a_max = -(-span // stride)  # ceil, on ints
  750|         steps = torch.arange(a_max, device=device, dtype=torch.long)
  751|         anchors = floor + phase[:, None] + steps[None, :] * stride
  752|         valid = anchors < t_valid
  753| 
  756|         count = (span - phase + stride - 1) // stride
  757|         last = floor + phase + (count - 1) * stride
  758|         return torch.where(valid, anchors, last[:, None]), valid
  759| 
  760|     def forward(
  761|         self,
  762|         y_st: torch.Tensor,
  763|         y_ph: torch.Tensor,
  764|         u_stream: torch.Tensor,
  765|         anchor_phase: Optional[Union[int, torch.Tensor]] = None,
  766|         anchor_stride: Optional[int] = None,
  767|     ) -> Dict[str, torch.Tensor]:
  806|         anchor_index, anchor_valid = self._build_anchor_index(
  807|             batch=int(y_st.shape[0]),
  808|             device=y_st.device,
  809|             anchor_phase=anchor_phase,
  810|             anchor_stride=anchor_stride,
  811|         )
  812| 
  817|         target = torch.cat([y_st, y_ph], dim=-1)
  827|         persistence = (
  828|             self._anchor_target_values(target, anchor_index)
  829|             if self.persistence_residual
  830|             else None
  831|         )
  832|         if self.target_gate is not None:
  833|             target = self.target_gate(target)
  834|         source = u_stream if self.source_gate is None else self.source_gate(u_stream)
  835| 
  836|         h_y = self.target_encoder(self.target_adapter(target))
  841|         h_u = self.encode_source_kv(source)
  842| 
  851|         mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(
  852|             h_y,
  853|             clock=self._prior_clock(u_stream) if self.prior_availability_input else None,
  854|         )
  855| 
  859|         query = (
  860|             torch.cat([mu_prior, logvar_prior], dim=-1)
  861|             if self.query_uses_logvar
  862|             else mu_prior
  863|         )
  864|         _, alpha, attended_heads = self.lag_attn(
  865|             self.query_proj(query), h_u, self.build_lag_mask(h_u.shape[1], h_u.device)
  866|         )
  867| 
  868|         mu_post, logvar_post = self.posterior_head(
  869|             h_y, attended_heads, mu_prior, raw_logvar_prior
  870|         )
  871|         z_prior, z_post = self._reparameterize_shared(
  872|             mu_prior, logvar_prior, mu_post, logvar_post
  873|         )
  874| 
  877|         with torch.no_grad():
  878|             mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
  879|             delta_mu_sat_frac = (
  880|                 (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
  881|             ).float().mean()
  882| 
  886|         gather_index = anchor_index[:, :, None].expand(-1, -1, self.d_z)
  887|         mu_base, logvar_base = self.decoder(
  888|             z_prior.gather(1, gather_index), persistence=persistence
  889|         )
  890|         mu_full, logvar_full = self.decoder(
  891|             z_post.gather(1, gather_index), persistence=persistence
  892|         )
  893| 
  897|         kld_btd = self.kld_tensor(
  898|             mu_prior=mu_prior,
  899|             logvar_prior=logvar_prior,
  900|             mu_post=mu_post,
  901|             logvar_post=logvar_post,
  902|         )
  903|         kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
  904|             kld_btd, alpha, head_structured=True
  905|         )
  906| 
  907|         outputs = {
  908|             "mu_prior": mu_prior,
  909|             "logvar_prior": logvar_prior,
  910|             "raw_logvar_prior": raw_logvar_prior,
  911|             "mu_post": mu_post,
  912|             "logvar_post": logvar_post,
  913|             "z_prior": z_prior,
  914|             "z_post": z_post,
  915|             "target_state": h_y,
  916|             "source_state": h_u,
  917|             "attended_source_heads": attended_heads,
  918|             "attn_weights": alpha,
  919|             "mu_base": mu_base,
  920|             "logvar_base": logvar_base,
  921|             "mu_full": mu_full,
  922|             "logvar_full": logvar_full,
  923|             "kld_per_t": kld_per_t,
  924|             "kld_per_t_per_head": kld_per_t_per_head,
  925|             "source_kl_lag_map": source_kl_lag_map,
  926|             "mu_prior_sat_frac": mu_prior_sat_frac,
  927|             "delta_mu_sat_frac": delta_mu_sat_frac,
  928|             "anchor_index": anchor_index,
  929|             "anchor_valid": anchor_valid,
  930|         }
  931|         if persistence is not None:
  937|             outputs["persistence"] = persistence
  938|         return outputs
  939| 
  941| __all__ = ["CAUSAL_ONLY_KEYWORDS", "FORWARDED_EXCLUSIONS", "CausalWarmupInputs"]
```

## 13. Causal feature-target mixin — `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` (`CausalFeatureForecastTarget`)

Block splits (36/36), anchor-floor refusal, channel-weight resolution and buffer, persistence admission and anchor-value gather, forecast-clock target gather, pooled validity, `compute_loss` override.

`teb_vae/lag_attn_cfs/nets/causal_feature_target.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   57| from __future__ import annotations
   58| 
   59| from typing import Any, Dict, Optional, Sequence, Tuple
   60| 
   61| import torch
   62| import torch.nn.functional as F
   63| 
   64| from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
   65| from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
   66| from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask
   67| 
   72| WARM_TERTILES = 3
   73| 
   79| WARM_BLOCK_FRACTION = 0.5
   80| 
   82| def pooled_scored_weight(
   83|     weight: torch.Tensor, shift: Optional[Sequence[int]]
   84| ) -> torch.Tensor:
  104|     if not shift:
  105|         return weight
  106|     low, high = min(0, min(shift)), max(0, max(shift))
  107|     if low == 0 and high == 0:
  108|         return weight
  111|     padded = F.pad(weight.unsqueeze(1), (-low, high), value=1.0)
  112|     pooled = -F.max_pool1d(-padded, kernel_size=high - low + 1, stride=1)
  113|     return pooled.squeeze(1)
  114| 
  116| class CausalFeatureForecastTarget(FeatureForecastTarget):
  141| 
  142|     target_warm_frac: float
  143|     target_novelty_frac: Optional[Tuple[float, ...]]
  144|     warm_tertile_id: torch.Tensor
  145|     novelty_tertile_id: torch.Tensor
  146|     source_block_warm_st: torch.Tensor
  147|     source_block_warm_ph: torch.Tensor
  148| 
  160|     TARGET_BLOCK_SPLIT: int = 36
  161| 
  175|     SOURCE_BLOCK_SPLIT: int = 36
  176| 
  177|     @staticmethod
  178|     def _check_anchor_floor(
  179|         warmup_period: int,
  180|         kept_warmup_steps: Sequence[int],
  181|         kept_align_delays: Sequence[int] = (),
  182|         target_forecast_shift: Sequence[int] = (),
  183|     ) -> None:
  232|         if not kept_warmup_steps:
  233|             return
  234|         waits = [int(step) for step in kept_warmup_steps]
  235|         shifts = [int(shift) for shift in kept_align_delays]
  236|         if shifts and len(shifts) != len(waits):
  237|             raise ValueError(
  238|                 f"kept_align_delays has {len(shifts)} entries against {len(waits)} kept warm-up "
  239|                 f"steps. Both are positional over the same surviving channels, so a length "
  240|                 f"mismatch would pair one channel's wait with another's shift and refuse -- or "
  241|                 f"admit -- a floor computed for a stream that does not exist."
  242|             )
  243|         fshifts = [int(shift) for shift in target_forecast_shift]
  244|         if fshifts and len(fshifts) != len(waits):
  245|             raise ValueError(
  246|                 f"target_forecast_shift has {len(fshifts)} entries against {len(waits)} kept "
  247|                 f"warm-up steps. Both are positional over the same surviving channels, so a "
  248|                 f"length mismatch would pair one channel's wait with another's clock and refuse "
  249|                 f"-- or admit -- a floor computed for a stream that does not exist."
  250|             )
  251| 
  252|         budget = max(waits)
  253|         required = budget - 1
  254|         binding = "the scored target's validity"
  255|         detail = (
  256|             f"the slowest kept target channel is honest only from step {budget}, and a forecast "
  257|             f"at anchor t reads target step t + 1 at the earliest"
  258|         )
  259|         consequence = (
  260|             "Below it the objective scores assumed pre-recording history as signal, on "
  261|             "coefficients normalised with constants that excluded exactly that region"
  262|         )
  263| 
  267|         if fshifts and any(fshifts):
  268|             scored = [wait - shift for wait, shift in zip(waits, fshifts)]
  269|             index = max(range(len(scored)), key=scored.__getitem__)
  270|             required = scored[index] - 1
  271|             detail = (
  272|                 f"kept target channel {index} is scored at stored step t + 1 "
  273|                 f"{fshifts[index]:+d} on the configured forecast clock and its own warm-up is "
  274|                 f"{waits[index]} steps, so its first horizon element is honest only from anchor "
  275|                 f"{required}"
  276|             )
  277| 
  281|         if shifts and max(shifts) > 0:
  282|             combined = [wait + shift for wait, shift in zip(waits, shifts)]
  283|             index = max(range(len(combined)), key=combined.__getitem__)
  284|             if combined[index] > required:
  285|                 required = combined[index]
  286|                 binding = "the shifted inputs' warmth"
  287|                 detail = (
  288|                     f"kept target channel {index} is gathered from step t - {shifts[index]} and "
  289|                     f"its own warm-up is {waits[index]} steps, so it is honest at the anchor only "
  290|                     f"from step {required}"
  291|                 )
  292|                 consequence = (
  293|                     "Below it the aligned channel vector claims one physical instant while an "
  294|                     "entry of it has not arrived, which is the whole property the shift applies "
  295|                     "the channels for"
  296|                 )
  297| 
  298|         if int(warmup_period) < required:
  299|             raise ValueError(
  300|                 f"warmup_period={int(warmup_period)} is below the anchor floor {binding} "
  301|                 f"requires: {detail}, so the floor must be at least {required}. {consequence} -- "
  302|                 f"with every shape correct and nothing reporting it."
  303|             )
  304| 
  361| 
  391| 
  408| 
  433| 
  434|     def _set_target_novelty(
  435|         self, *, target_novelty_frac: Optional[Sequence[float]]
  436|     ) -> None:
  457|         self.target_novelty_frac = (
  458|             None
  459|             if target_novelty_frac is None
  460|             else tuple(float(share) for share in target_novelty_frac)
  461|         )
  462| 
  463|     def _set_channel_weights(self, *, target_weight_st: float, target_weight_ph: float) -> None:
  477|         self.target_weight_st = float(target_weight_st)
  478|         self.target_weight_ph = float(target_weight_ph)
  479| 
  480|     def _register_channel_weights(self) -> None:
  493|         declared = (
  494|             torch.arange(self.c_y)
  495|             if self.target_gate is None
  496|             else self.target_gate.keep_index.cpu()
  497|         )
  498|         self.register_buffer(
  499|             "target_channel_weight",
  500|             self._resolve_channel_weights(
  501|                 declared.tolist(),
  502|                 weight_st=self.target_weight_st,
  503|                 weight_ph=self.target_weight_ph,
  504|             ),
  505|             persistent=False,
  506|         )
  507| 
  508|     @classmethod
  509|     def _resolve_channel_weights(
  510|         cls,
  511|         keep_index: Sequence[int],
  512|         *,
  513|         weight_st: float,
  514|         weight_ph: float,
  515|     ) -> torch.Tensor:
  549|         for name, value in (("target_weight_st", weight_st), ("target_weight_ph", weight_ph)):
  550|             if not float(value) >= 0.0:  # catches NaN as well as negatives
  551|                 raise ValueError(f"{name} must be >= 0 and not NaN, got {value!r}")
  552| 
  553|         declared = torch.as_tensor(list(keep_index), dtype=torch.long)
  554|         weights = torch.where(
  555|             declared < cls.TARGET_BLOCK_SPLIT,
  556|             torch.tensor(float(weight_st)),
  557|             torch.tensor(float(weight_ph)),
  558|         ).to(torch.float32)
  559| 
  560|         total = float(weights.sum())
  561|         if total <= 0.0:
  562|             raise ValueError(
  563|                 "target_weight_st and target_weight_ph are both zero, which would leave the "
  564|                 "objective with no reconstruction term; at least one block must carry weight"
  565|             )
  566|         return weights * (float(weights.numel()) / total)
  567| 
  602| 
  606|     def _check_persistence_target(self) -> None:
  607|         pass  # (docstring-only body)
  620| 
  621|     def _build_forecast_target(
  622|         self, target_features: torch.Tensor, anchors: Optional[torch.Tensor] = None
  623|     ) -> torch.Tensor:
  654|         shift = getattr(self, "target_forecast_shift", None)
  655|         if not shift or not any(shift):
  656|             return super()._build_forecast_target(target_features, anchors)
  657| 
  660|         if target_features.dim() != 3:
  661|             raise ValueError(
  662|                 f"target stream must be 3-D (B, T, c_y), got shape {tuple(target_features.shape)}"
  663|             )
  664|         if target_features.size(1) != self.geometry.t:
  665|             raise ValueError(
  666|                 f"target stream length {target_features.size(1)} != geometry.t "
  667|                 f"{self.geometry.t}; this geometry assumes the trimmed loader "
  668|                 f"(trim_minutes: 1.0 -> T = {self.geometry.t} decimated steps), so a mismatch "
  669|                 f"means the loader ran at a different trim_minutes"
  670|             )
  671|         if target_features.size(2) != self.c_y:
  672|             raise ValueError(
  673|                 f"target stream has {target_features.size(2)} channels but the model declares "
  674|                 f"c_y={self.c_y}; the surviving-channel index is positional into the declared "
  675|                 f"width, so a mismatch would gather the wrong channels rather than fail"
  676|             )
  677| 
  678|         gathered = (
  679|             target_features
  680|             if self.target_gate is None
  681|             else torch.index_select(target_features, -1, self.target_gate.keep_index)
  682|         )
  683|         if anchors is None:
  688|             anchors = (
  689|                 torch.arange(self.anchor_ceiling, device=gathered.device)
  690|                 .unsqueeze(0)
  691|                 .expand(gathered.shape[0], -1)
  692|             )
  693| 
  694|         batch, channels = gathered.shape[0], gathered.shape[-1]
  695|         steps = torch.arange(self.horizon, device=gathered.device)
  698|         base_index = anchors.to(torch.long)[:, :, None] + 1 + steps[None, None, :]
  699|         out = gathered.new_empty(batch, anchors.shape[1], self.horizon, channels)
  700|         for value in sorted(set(shift)):
  701|             columns = torch.tensor(
  702|                 [index for index, s in enumerate(shift) if s == value],
  703|                 dtype=torch.long,
  704|                 device=gathered.device,
  705|             )
  706|             block = torch.index_select(gathered, -1, columns)
  707|             time_index = (
  708|                 (base_index + int(value))
  709|                 .reshape(batch, -1, 1)
  710|                 .expand(-1, -1, columns.numel())
  711|             )
  712|             window = block.gather(1, time_index).reshape(
  713|                 batch, anchors.shape[1], self.horizon, columns.numel()
  714|             )
  715|             out.index_copy_(-1, columns, window)
  716|         return out
  717| 
  718|     def scored_weight(self, weight: torch.Tensor) -> torch.Tensor:
  749|         return pooled_scored_weight(
  750|             weight, getattr(self, "target_forecast_shift", None)
  751|         )
  752| 
  753|     def compute_loss(
  754|         self,
  755|         forward_outputs: Dict[str, torch.Tensor],
  756|         target_features: torch.Tensor,
  757|         *,
  758|         weight: torch.Tensor,
  759|         beta: float = 1.0,
  760|         beta_prior: float = 0.0,
  761|         lambda_full: float = 1.0,
  762|         lambda_base: float = 1.0,
  763|         likelihood: str = "gaussian_nll",
  764|         free_bits: float = 0.0,
  765|         lambda_ms: float = 0.0,
  766|         lambda_deriv: float = 0.0,
  767|         lambda_boundary: float = 0.0,
  768|     ) -> Dict[str, Any]:
  784|         return super().compute_loss(
  785|             forward_outputs,
  786|             target_features,
  787|             weight=self.scored_weight(weight),
  788|             beta=beta,
  789|             beta_prior=beta_prior,
  790|             lambda_full=lambda_full,
  791|             lambda_base=lambda_base,
  792|             likelihood=likelihood,
  793|             free_bits=free_bits,
  794|             lambda_ms=lambda_ms,
  795|             lambda_deriv=lambda_deriv,
  796|             lambda_boundary=lambda_boundary,
  797|         )
  798| 
  799|     def _anchor_target_values(
  800|         self, target_features: torch.Tensor, anchors: torch.Tensor
  801|     ) -> torch.Tensor:
  836|         gathered = (
  837|             target_features
  838|             if self.target_gate is None
  839|             else torch.index_select(target_features, -1, self.target_gate.keep_index)
  840|         )
  841|         shift = getattr(self, "target_forecast_shift", None)
  842|         if not shift or min(shift) >= 0:
  843|             index = anchors.to(torch.long)[:, :, None].expand(-1, -1, gathered.shape[-1])
  844|             return gathered.gather(1, index)
  847|         offsets = torch.tensor(
  848|             [min(int(s), 0) for s in shift], dtype=torch.long, device=gathered.device
  849|         )
  850|         index = anchors.to(torch.long)[:, :, None] + offsets[None, None, :]
  851|         return gathered.gather(1, index)
  852| 
  902| 
  939| 
 1018| 
```

## 14. Feature-target models

### 14.1 `SeqVaeLagAttnCfs` — `teb_vae/lag_attn_cfs/nets/model.py`

`teb_vae/lag_attn_cfs/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   28| from __future__ import annotations
   29| 
   30| from typing import Optional, Sequence, Tuple
   31| 
   32| from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
   33| from teb_vae.lag_attn_cfs.nets.causal_inputs import (
   34|     FORWARDED_EXCLUSIONS,
   35|     CausalWarmupInputs,
   36| )
   37| from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
   38| 
   40| class SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws):
   53| 
   54|     def __init__(
   55|         self,
   56|         *,
   57|         sequence_length: int = 300,
   58|         d_model: int = 128,
   59|         d_z: int = 48,
   60|         horizon: int = 30,
   61|         raw_per_step: int = 16,
   62|         warmup_period: int = 134,
   63|         c_y: int = 102,
   64|         c_u: int = 51,
   65|         use_up_st: bool = True,
   66|         max_lag: int = 90,
   67|         num_heads: int = 4,
   68|         d_head: int = 32,
   69|         lstm_layers: int = 2,
   70|         dropout: float = 0.1,
   71|         decoder_hidden: int = 128,
   72|         decoder_out_channels: Optional[int] = None,
   73|         horizon_depth: int = 2,
   74|         horizon_kernel: int = 3,
   75|         horizon_film: bool = False,
   76|         horizon_attention_blocks: int = 0,
   77|         horizon_embed_std: float = 0.02,
   78|         head_init_calibration: bool = False,
   79|         a_head_gain: float = 1.0,
   80|         encoder_extra_dilations: Tuple[int, ...] = (),
   81|         encoder_extra_kernel: int = 15,
   82|         conv_norm_groups: Optional[int] = None,
   83|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
   84|         mu_scale: float = 5.0,
   85|         delta_mu_scale: float = 3.0,
   86|         delta_logvar_scale: float = 2.0,
   87|         posterior_logvar_mode: str = "residual",
   88|         source_dropout: Optional[float] = None,
   89|         lag_kv_source: str = "encoder",
   90|         use_entmax: bool = False,
   91|         attention_grad_checkpoint: bool = False,
   92|         lag_bias_init: str = "normal",
   93|         alibi_slope_scale: float = 1.0,
   94|         query_uses_logvar: bool = False,
   95|         prior_availability_input: bool = False,
   96|         causal_norm: bool = False,
   97|         coverage_floor: float = 0.9,
   98|         base_decode: str = "sample",
   99|         persistence_residual: bool = False,
  100|         horizon_weight_halflife_steps: Optional[float] = None,
  101|         target_keep_index: Optional[Sequence[int]] = None,
  102|         target_warmup_steps: Optional[Sequence[int]] = None,
  103|         source_keep_index: Optional[Sequence[int]] = None,
  104|         source_warmup_steps: Optional[Sequence[int]] = None,
  105|         target_align_delays: Optional[Sequence[int]] = None,
  106|         source_align_delays: Optional[Sequence[int]] = None,
  107|         anchor_stride: int = 1,
  108|         lag_floor: int = 0,
  109|         target_weight_st: float = 1.0,
  110|         target_weight_ph: float = 1.0,
  111|         target_novelty_frac: Optional[Sequence[float]] = None,
  112|         target_forecast_shift: Optional[Sequence[int]] = None,
  113|         init_weights: bool = True,
  114|     ) -> None:
  173|         forwarded = {
  174|             name: value
  175|             for name, value in locals().items()
  176|             if name not in FORWARDED_EXCLUSIONS
  177|         }
  178| 
  180|         self._set_causal_inputs(
  181|             horizon=horizon,
  182|             target_keep_index=target_keep_index,
  183|             target_warmup_steps=target_warmup_steps,
  184|             source_keep_index=source_keep_index,
  185|             source_warmup_steps=source_warmup_steps,
  186|             anchor_stride=anchor_stride,
  187|             lag_floor=lag_floor,
  188|             target_forecast_shift=target_forecast_shift,
  189|         )
  192|         self._set_channel_weights(
  193|             target_weight_st=target_weight_st, target_weight_ph=target_weight_ph
  194|         )
  198|         self._set_target_novelty(target_novelty_frac=target_novelty_frac)
  199| 
  205|         super().__init__(
  206|             **forwarded,
  207|             target_delays=target_align_delays,
  208|             source_delays=source_align_delays,
  209|         )
  210| 
  212|         self._validate_causal_geometry()
  215|         self._register_channel_weights()
```

### 14.2 `SeqVaeLagAttnTrfCfs` — `teb_vae/lag_attn_transformer_cfs/nets/model.py`

`teb_vae/lag_attn_transformer_cfs/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   44| from __future__ import annotations
   45| 
   46| from typing import Optional, Sequence, Tuple
   47| 
   48| from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
   49| from teb_vae.lag_attn_cfs.nets.causal_inputs import (
   50|     FORWARDED_EXCLUSIONS,
   51|     CausalWarmupInputs,
   52| )
   53| from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
   54| 
   56| class SeqVaeLagAttnTrfCfs(
   57|     CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws
   58| ):
   74| 
   75|     def __init__(
   76|         self,
   77|         *,
   78|         sequence_length: int = 300,
   79|         d_model: int = 128,
   80|         d_z: int = 48,
   81|         horizon: int = 30,
   82|         raw_per_step: int = 16,
   83|         warmup_period: int = 134,
   84|         c_y: int = 102,
   85|         c_u: int = 51,
   86|         use_up_st: bool = True,
   87|         max_lag: int = 90,
   88|         num_heads: int = 4,
   89|         d_head: int = 32,
   90|         dropout: float = 0.1,
   91|         decoder_hidden: int = 128,
   92|         horizon_depth: int = 2,
   93|         horizon_kernel: int = 3,
   94|         horizon_film: bool = False,
   95|         horizon_attention_blocks: int = 0,
   96|         horizon_embed_std: float = 0.02,
   97|         head_init_calibration: bool = False,
   98|         a_head_gain: float = 1.0,
   99|         encoder_conv_kernels: Sequence[int] = (5, 9),
  100|         encoder_conv_dilations: Sequence[int] = (1, 2),
  101|         encoder_num_heads: int = 4,
  102|         encoder_d_ff: int = 256,
  103|         target_attention_blocks: int = 4,
  104|         source_attention_blocks: int = 3,
  105|         source_attention_window: Optional[int] = 16,
  106|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  107|         mu_scale: float = 5.0,
  108|         delta_mu_scale: float = 3.0,
  109|         delta_logvar_scale: float = 2.0,
  110|         posterior_logvar_mode: str = "residual",
  111|         source_dropout: Optional[float] = None,
  112|         lag_kv_source: str = "encoder",
  113|         use_entmax: bool = False,
  114|         attention_grad_checkpoint: bool = False,
  115|         lag_bias_init: str = "normal",
  116|         alibi_slope_scale: float = 1.0,
  117|         query_uses_logvar: bool = False,
  118|         prior_availability_input: bool = False,
  119|         coverage_floor: float = 0.9,
  120|         base_decode: str = "sample",
  121|         persistence_residual: bool = False,
  122|         horizon_weight_halflife_steps: Optional[float] = None,
  123|         target_keep_index: Optional[Sequence[int]] = None,
  124|         target_warmup_steps: Optional[Sequence[int]] = None,
  125|         source_keep_index: Optional[Sequence[int]] = None,
  126|         source_warmup_steps: Optional[Sequence[int]] = None,
  127|         target_align_delays: Optional[Sequence[int]] = None,
  128|         source_align_delays: Optional[Sequence[int]] = None,
  129|         anchor_stride: int = 1,
  130|         lag_floor: int = 0,
  131|         target_weight_st: float = 1.0,
  132|         target_weight_ph: float = 1.0,
  133|         target_novelty_frac: Optional[Sequence[float]] = None,
  134|         target_forecast_shift: Optional[Sequence[int]] = None,
  135|         init_weights: bool = True,
  136|     ) -> None:
  198|         forwarded = {
  199|             name: value
  200|             for name, value in locals().items()
  201|             if name not in FORWARDED_EXCLUSIONS
  202|         }
  203| 
  205|         self._set_causal_inputs(
  206|             horizon=horizon,
  207|             target_keep_index=target_keep_index,
  208|             target_warmup_steps=target_warmup_steps,
  209|             source_keep_index=source_keep_index,
  210|             source_warmup_steps=source_warmup_steps,
  211|             anchor_stride=anchor_stride,
  212|             lag_floor=lag_floor,
  213|             target_forecast_shift=target_forecast_shift,
  214|         )
  217|         self._set_channel_weights(
  218|             target_weight_st=target_weight_st, target_weight_ph=target_weight_ph
  219|         )
  223|         self._set_target_novelty(target_novelty_frac=target_novelty_frac)
  224| 
  230|         super().__init__(
  231|             **forwarded,
  232|             target_delays=target_align_delays,
  233|             source_delays=source_align_delays,
  234|         )
  235| 
  237|         self._validate_causal_geometry()
  240|         self._register_channel_weights()
```

## 15. Raw-target models

### 15.1 `CausalRawInputs` — `teb_vae/lag_attn_crws/nets/causal_raw_inputs.py`

Extends `CausalWarmupInputs`; binds the source-side constants and readouts from `CausalFeatureForecastTarget`; overrides `_check_anchor_floor` (policy wording, 3 args) and `compute_loss` (anchored raw gather `gather_anchored_future_target`, `block_width=16`, no channel weight).

`teb_vae/lag_attn_crws/nets/causal_raw_inputs.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   48| from __future__ import annotations
   49| 
   50| from typing import Any, Dict, Optional, Sequence
   51| 
   52| import torch
   53| 
   54| from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
   55| from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
   56| from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
   57| from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_raw_objective
   58| from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target
   59| 
   61| def gather_anchored_future_target(
   62|     fhr_raw: torch.Tensor,
   63|     geometry: TrimmedRawGeometry,
   64|     anchors: torch.Tensor,
   65|     *,
   66|     future_index: torch.Tensor,
   67| ) -> torch.Tensor:
  108|     if fhr_raw.dim() != 2:
  109|         raise ValueError(f"fhr_raw must be 2-D (B, L_raw), got shape {tuple(fhr_raw.shape)}")
  110|     if fhr_raw.size(1) != geometry.raw_len:
  111|         raise ValueError(
  112|             f"fhr_raw length {fhr_raw.size(1)} != geometry.raw_len {geometry.raw_len}; "
  113|             "this geometry assumes the loader's symmetric trim has already been applied "
  114|             "(trim_minutes: 1.0 -> 4800 raw samples), so a mismatch means the loader ran "
  115|             "at a different trim_minutes than the geometry was built for"
  116|         )
  117|     if anchors.dim() != 2:
  118|         raise ValueError(f"anchors must be 2-D (B, A), got shape {tuple(anchors.shape)}")
  119| 
  120|     index = anchors.to(device=fhr_raw.device, dtype=torch.long)
  121|     outside = (index < 0) | (index >= geometry.t_valid)
  122|     if bool(outside.any()):
  123|         offending = int(index[outside][0])
  124|         raise ValueError(
  125|             f"anchor {offending} is outside [0, T_valid) = [0, {geometry.t_valid}); the tail "
  126|             f"{geometry.horizon} anchors have no fully observed forecast window, and a negative "
  127|             f"index would wrap to a legal window rather than raising"
  128|         )
  129| 
  130|     batch, count = int(index.shape[0]), int(index.shape[1])
  132|     windows = future_index.to(fhr_raw.device)[index]
  133|     gathered = fhr_raw.gather(1, windows.reshape(batch, -1))
  134|     return gathered.reshape(batch, count, geometry.horizon, geometry.r)
  135| 
  137| class CausalRawInputs(CausalWarmupInputs):
  166| 
  167|     source_block_warm_st: torch.Tensor
  168|     source_block_warm_ph: torch.Tensor
  169| 
  173|     SOURCE_BLOCK_SPLIT = CausalFeatureForecastTarget.SOURCE_BLOCK_SPLIT
  174|     TARGET_BLOCK_SPLIT = CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT
  175| 
  177|     _resolve_block_warm_steps = staticmethod(
  178|         CausalFeatureForecastTarget._resolve_block_warm_steps
  179|     )
  180| 
  184|     _anchors_per_sample = CausalFeatureForecastTarget._anchors_per_sample
  185|     _source_lag_warmth = CausalFeatureForecastTarget._source_lag_warmth
  186| 
  190|     @staticmethod
  191|     def _check_anchor_floor(
  192|         warmup_period: int,
  193|         kept_warmup_steps: Sequence[int],
  194|         kept_align_delays: Sequence[int] = (),
  195|     ) -> None:
  261|         if not kept_warmup_steps:
  262|             return
  263|         waits = [int(step) for step in kept_warmup_steps]
  264|         shifts = [int(shift) for shift in kept_align_delays]
  265|         if shifts and len(shifts) != len(waits):
  266|             raise ValueError(
  267|                 f"kept_align_delays has {len(shifts)} entries against {len(waits)} kept warm-up "
  268|                 f"steps. Both are positional over the same surviving channels, so a length "
  269|                 f"mismatch would pair one channel's wait with another's shift and refuse -- or "
  270|                 f"admit -- a floor computed for a stream that does not exist."
  271|             )
  272| 
  273|         budget = max(waits)
  274|         required = budget - 1
  275|         binding = (
  276|             f"every kept TARGET-STREAM input channel is warm by the first forecast step. The "
  277|             f"slowest kept target-stream channel is honest only from step {budget}, and a forecast "
  278|             f"at anchor t covers target steps from t + 1 onwards"
  279|         )
  280| 
  284|         if shifts and max(shifts) > 0:
  285|             combined = [wait + shift for wait, shift in zip(waits, shifts)]
  286|             index = max(range(len(combined)), key=combined.__getitem__)
  287|             if combined[index] > required:
  288|                 required = combined[index]
  289|                 binding = (
  290|                     f"every kept TARGET-STREAM input channel is warm AT THE ANCHOR once those "
  291|                     f"inputs are shifted onto a common clock -- below that the aligned channel "
  292|                     f"vector claims one physical instant while an entry of it has not arrived, "
  293|                     f"which is the whole property the shift was applied for. Kept target-stream "
  294|                     f"channel {index} is gathered from step t - {shifts[index]} and waits "
  295|                     f"{waits[index]} steps of its own, so it is honest at the anchor only from "
  296|                     f"step {combined[index]}"
  297|                 )
  298| 
  299|         if int(warmup_period) < required:
  300|             raise ValueError(
  301|                 f"warmup_period={int(warmup_period)} is below this cell's declared input-warmth "
  302|                 f"policy: {binding}, so the floor must be at least {required}. Two things this "
  303|                 f"does not say. The policy is over the gated target stream alone: the source "
  304|                 f"stream is never gated for its warm-up, and lag attention reads it BACK from the "
  305|                 f"anchor, so its lagged reads reach steps far colder than this floor by design -- "
  306|                 f"measured by source_lag_warmth_frac_st / _ph rather than refused here, so do not "
  307|                 f"raise the floor to cover them. And the raw target is honest at every step, so a "
  308|                 f"lower floor would not corrupt the objective -- it would decode anchors whose "
  309|                 f"inputs are still partly pre-recording history, which is a different claim about "
  310|                 f"the run rather than a wrong number in it."
  311|             )
  312| 
  382| 
  386|     def compute_loss(
  387|         self,
  388|         forward_outputs: Dict[str, torch.Tensor],
  389|         fhr_raw: torch.Tensor,
  390|         *,
  391|         weight: torch.Tensor,
  392|         beta: float = 1.0,
  393|         beta_prior: float = 0.0,
  394|         lambda_full: float = 1.0,
  395|         lambda_base: float = 1.0,
  396|         likelihood: str = "gaussian_nll",
  397|         free_bits: float = 0.0,
  398|         lambda_ms: float = 0.0,
  399|         lambda_deriv: float = 0.0,
  400|         lambda_boundary: float = 0.0,
  401|     ) -> Dict[str, Any]:
  454|         anchors: Optional[torch.Tensor] = forward_outputs.get("anchor_index")
  455|         target = (
  456|             build_future_target(fhr_raw, self.geometry, future_index=self.future_index)
  457|             if anchors is None
  458|             else gather_anchored_future_target(
  459|                 fhr_raw, self.geometry, anchors, future_index=self.future_index
  460|             )
  461|         )
  462|         result = compute_raw_objective(
  463|             forward_outputs,
  464|             target,
  465|             weight=weight,
  466|             geometry=self.geometry,
  468|             block_width=self.geometry.r,
  469|             coverage_floor=self.coverage_floor,
  470|             logvar_clamp=self.logvar_clamp,
  471|             beta=beta,
  472|             beta_prior=beta_prior,
  473|             lambda_full=lambda_full,
  474|             lambda_base=lambda_base,
  475|             likelihood=likelihood,
  476|             free_bits=free_bits,
  477|             lambda_ms=lambda_ms,
  478|             lambda_deriv=lambda_deriv,
  479|             lambda_boundary=lambda_boundary,
  484|             horizon_weight=getattr(self, "horizon_weight", None),
  485|         )
  488|         result["metrics"]["anchors_per_sample"] = self._anchors_per_sample(
  489|             forward_outputs, target
  490|         )
  491|         result["metrics"].update(self._source_lag_warmth(forward_outputs, target))
  492|         return result
  493| 
  495| __all__ = ["CausalRawInputs", "gather_anchored_future_target"]
```

### 15.2 `SeqVaeLagAttnCrws` — `teb_vae/lag_attn_crws/nets/model.py`

`teb_vae/lag_attn_crws/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   44| from __future__ import annotations
   45| 
   46| from typing import Optional, Sequence, Tuple
   47| 
   48| from teb_vae.lag_attn_cfs.nets.causal_inputs import FORWARDED_EXCLUSIONS
   49| from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
   50| from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
   51| 
   53| class SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws):
   82| 
   83|     def __init__(
   84|         self,
   85|         *,
   86|         sequence_length: int = 300,
   87|         d_model: int = 128,
   88|         d_z: int = 48,
   89|         horizon: int = 30,
   90|         raw_per_step: int = 16,
   91|         warmup_period: int = 134,
   92|         c_y: int = 102,
   93|         c_u: int = 51,
   94|         use_up_st: bool = True,
   95|         max_lag: int = 90,
   96|         num_heads: int = 4,
   97|         d_head: int = 32,
   98|         lstm_layers: int = 2,
   99|         dropout: float = 0.1,
  100|         decoder_hidden: int = 128,
  101|         decoder_out_channels: Optional[int] = None,
  102|         horizon_depth: int = 2,
  103|         horizon_kernel: int = 3,
  104|         horizon_film: bool = False,
  105|         horizon_attention_blocks: int = 0,
  106|         horizon_embed_std: float = 0.02,
  107|         head_init_calibration: bool = False,
  108|         a_head_gain: float = 1.0,
  109|         encoder_extra_dilations: Tuple[int, ...] = (),
  110|         encoder_extra_kernel: int = 15,
  111|         conv_norm_groups: Optional[int] = None,
  112|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  113|         mu_scale: float = 5.0,
  114|         delta_mu_scale: float = 3.0,
  115|         delta_logvar_scale: float = 2.0,
  116|         posterior_logvar_mode: str = "residual",
  117|         source_dropout: Optional[float] = None,
  118|         lag_kv_source: str = "encoder",
  119|         use_entmax: bool = False,
  120|         attention_grad_checkpoint: bool = False,
  121|         lag_bias_init: str = "normal",
  122|         alibi_slope_scale: float = 1.0,
  123|         query_uses_logvar: bool = False,
  124|         prior_availability_input: bool = False,
  125|         causal_norm: bool = False,
  126|         coverage_floor: float = 0.9,
  127|         base_decode: str = "sample",
  128|         horizon_weight_halflife_steps: Optional[float] = None,
  129|         target_keep_index: Optional[Sequence[int]] = None,
  130|         target_warmup_steps: Optional[Sequence[int]] = None,
  131|         source_keep_index: Optional[Sequence[int]] = None,
  132|         source_warmup_steps: Optional[Sequence[int]] = None,
  133|         target_align_delays: Optional[Sequence[int]] = None,
  134|         source_align_delays: Optional[Sequence[int]] = None,
  135|         anchor_stride: int = 1,
  136|         lag_floor: int = 0,
  137|         init_weights: bool = True,
  138|     ) -> None:
  181|         forwarded = {
  182|             name: value
  183|             for name, value in locals().items()
  184|             if name not in FORWARDED_EXCLUSIONS
  185|         }
  186| 
  188|         self._set_causal_inputs(
  189|             horizon=horizon,
  190|             target_keep_index=target_keep_index,
  191|             target_warmup_steps=target_warmup_steps,
  192|             source_keep_index=source_keep_index,
  193|             source_warmup_steps=source_warmup_steps,
  194|             anchor_stride=anchor_stride,
  195|             lag_floor=lag_floor,
  196|         )
  197| 
  203|         super().__init__(
  204|             **forwarded,
  205|             target_delays=target_align_delays,
  206|             source_delays=source_align_delays,
  207|         )
  208| 
  210|         self._validate_causal_geometry()
```

### 15.3 `SeqVaeLagAttnTrfCrws` — `teb_vae/lag_attn_transformer_crws/nets/model.py`

`teb_vae/lag_attn_transformer_crws/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   59| from __future__ import annotations
   60| 
   61| from typing import Optional, Sequence, Tuple
   62| 
   63| from teb_vae.lag_attn_cfs.nets.causal_inputs import FORWARDED_EXCLUSIONS
   64| from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
   65| from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
   66| 
   68| class SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws):
   84| 
   85|     def __init__(
   86|         self,
   87|         *,
   88|         sequence_length: int = 300,
   89|         d_model: int = 128,
   90|         d_z: int = 48,
   91|         horizon: int = 30,
   92|         raw_per_step: int = 16,
   93|         warmup_period: int = 134,
   94|         c_y: int = 102,
   95|         c_u: int = 51,
   96|         use_up_st: bool = True,
   97|         max_lag: int = 90,
   98|         num_heads: int = 4,
   99|         d_head: int = 32,
  100|         dropout: float = 0.1,
  101|         decoder_hidden: int = 128,
  102|         horizon_depth: int = 2,
  103|         horizon_kernel: int = 3,
  104|         horizon_film: bool = False,
  105|         horizon_attention_blocks: int = 0,
  106|         horizon_embed_std: float = 0.02,
  107|         head_init_calibration: bool = False,
  108|         a_head_gain: float = 1.0,
  109|         encoder_conv_kernels: Sequence[int] = (5, 9),
  110|         encoder_conv_dilations: Sequence[int] = (1, 2),
  111|         encoder_num_heads: int = 4,
  112|         encoder_d_ff: int = 256,
  113|         target_attention_blocks: int = 4,
  114|         source_attention_blocks: int = 3,
  115|         source_attention_window: Optional[int] = 16,
  116|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  117|         mu_scale: float = 5.0,
  118|         delta_mu_scale: float = 3.0,
  119|         delta_logvar_scale: float = 2.0,
  120|         posterior_logvar_mode: str = "residual",
  121|         source_dropout: Optional[float] = None,
  122|         lag_kv_source: str = "encoder",
  123|         use_entmax: bool = False,
  124|         attention_grad_checkpoint: bool = False,
  125|         lag_bias_init: str = "normal",
  126|         alibi_slope_scale: float = 1.0,
  127|         query_uses_logvar: bool = False,
  128|         prior_availability_input: bool = False,
  129|         coverage_floor: float = 0.9,
  130|         base_decode: str = "sample",
  131|         horizon_weight_halflife_steps: Optional[float] = None,
  132|         target_keep_index: Optional[Sequence[int]] = None,
  133|         target_warmup_steps: Optional[Sequence[int]] = None,
  134|         source_keep_index: Optional[Sequence[int]] = None,
  135|         source_warmup_steps: Optional[Sequence[int]] = None,
  136|         target_align_delays: Optional[Sequence[int]] = None,
  137|         source_align_delays: Optional[Sequence[int]] = None,
  138|         anchor_stride: int = 1,
  139|         lag_floor: int = 0,
  140|         init_weights: bool = True,
  141|     ) -> None:
  186|         forwarded = {
  187|             name: value
  188|             for name, value in locals().items()
  189|             if name not in FORWARDED_EXCLUSIONS
  190|         }
  191| 
  193|         self._set_causal_inputs(
  194|             horizon=horizon,
  195|             target_keep_index=target_keep_index,
  196|             target_warmup_steps=target_warmup_steps,
  197|             source_keep_index=source_keep_index,
  198|             source_warmup_steps=source_warmup_steps,
  199|             anchor_stride=anchor_stride,
  200|             lag_floor=lag_floor,
  201|         )
  202| 
  208|         super().__init__(
  209|             **forwarded,
  210|             target_delays=target_align_delays,
  211|             source_delays=source_align_delays,
  212|         )
  213| 
  215|         self._validate_causal_geometry()
```
