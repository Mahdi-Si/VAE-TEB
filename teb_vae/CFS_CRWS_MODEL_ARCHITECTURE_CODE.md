# Model architectures and network code — lag_attn_cfs / lag_attn_transformer_cfs / lag_attn_crws / lag_attn_transformer_crws

This file is the companion to [CFS_CRWS_MODELS_REFERENCE.md](CFS_CRWS_MODELS_REFERENCE.md), which covers the data pipeline, configs, objective, training and evaluation. This file holds only the **network architecture**: every `nn.Module`, mixin and constructor that the four models are built from, in dependency order, with the source code. It also holds the module trees and forward shape traces of the four models, built from their `configs/default.yaml`.

How to read the code blocks:

- Each block is the source file with **docstrings and full-line comments removed**. Every remaining line is verbatim. The prefix `NNNN|` is the line number in the source file. Blank-line runs are collapsed.
- Some code is omitted on purpose because it computes metrics or readouts, not the architecture: `FeatureForecastTarget._resolved_forecast_gaps` and `_forecast_gaps_from_mask` (`teb_vae/lag_attn_fs/nets/feature_target.py`); `CausalFeatureForecastTarget._resolve_target_warm_frac`, `_rank_tertiles`, `_resolve_warm_tertiles`, `_resolve_novelty_tertiles`, `_resolve_block_warm_steps`, `_gap_by_kept_channel`, `_anchors_per_sample`, `_source_lag_warmth` and `_resolved_forecast_gaps` (`teb_vae/lag_attn_cfs/nets/causal_feature_target.py`); `CausalWarmupInputs._resolve_warmup_readout_constants` (`causal_inputs.py`); `CausalRawInputs._resolve_warmup_readout_constants` (`causal_raw_inputs.py`). These files are also not reproduced: `teb_vae/lag_attn_rws/nets/losses.py`, `raw_masks.py` and `raw_targets.py` (the objective; the companion gives the formulas), `controls.py` (source-null and permutation controls) and `lag_report.py`.
- The trees and shape traces come from instantiating each class with the kwargs that the trainer derives from `default.yaml`: the `inspect.signature` sweep plus `warmup_model_kwargs(resolve_warmup_budget(...))`, through each trainer's `_build_model_kwargs`. The shard paths point at committed fixtures that carry the filter-bank metadata which the budget resolver reads. The two CFS cells use `teb_vae/lag_attn/tests/fixtures/tiny_shard_causal_int.hdf5` (`integer_harmonic_v1`). The two CRWS cells use `tiny_shard_causal.hdf5` (`ratio_power_v0`), which matches their declared widths $c_y=102$, $c_u=51$.
- Regenerate the code blocks, trees and traces from the executable source when the network code, the constructors or the `default.yaml` files change. Do not hand-edit numbered code excerpts.

---

## 0. Architecture at a glance

### 0.1 One pipeline, two axes

All four models share one forward (`CausalWarmupInputs.forward`, §12). They differ along two axes: the **encoder** (conv-LSTM `CausalConvLstmEncoder` or conv-Transformer `CausalConvTransformerEncoder`) and the **target** ($C_{\mathrm{keep}}$ kept one-sided feature channels, or 16 raw FHR samples per horizon step).

```
 y_st (B,T,36) ─┐                                         u_stream (B,T,c_u)
 y_ph (B,T,·)  ─┴─ cat ─► target (B,T,c_y)                     │
                             │                                  │
                      target_gate: ChannelGate           source_gate: ChannelGate
                      gather keep_index + delay d_c      gather keep_index + delay d_c
                      (B,T,C_keep_y)                     (B,T,C_keep_u)
                             │                                  │
                      target_adapter:                    source_adapter:
                      AvailabilityInputAdapter           AvailabilityInputAdapter
                      mask m_t,c=1[t>=W'_c+d_c]          (B,T,d_model)
                      (B,T,d_model)                             │
                             │                           encode_source_kv  (lag_kv_source)
                      target_encoder                       'conv_stem' -> source_kv_stem
                      conv-LSTM | conv-Transformer         'encoder'   -> source_encoder
                      h_y (B,T,d_model)                    'adapter'   -> identity
                             │                           h_u (B,T,d_model) = keys AND values
                             │                                  │
              prior_head: FullLatentPriorHead(h_y, clock)       │   clock = encode_source_kv(gate(0))
              mu_p, logvar_p, raw_logvar_p (B,T,d_z)            │   (B=1, detached, eval-mode)
                             │                                  │
                query_proj(mu_p) (B,T,d_model) ──► lag_attn: LagCrossAttention(q, h_u, mask)
                                                 mask (T,L), or (B,T,L) under source_quality_mask
                                                 alpha (B,T,M,L), attended_heads (B,T,M,d_head)
                             │                                  │
              posterior_head: PosteriorHead(h_y, attended_heads, mu_p, raw_logvar_p[, gate])
              mu_q = mu_p + g_t * 3 tanh(./3), logvar_q (independent head)   (B,T,d_z)
              g_t = event gate (posterior_event_gate), or 1
                             │
              _reparameterize_shared: eps ~ N(0,I); z_q = mu_q + sigma_q eps; z_p = mu_p (base_decode=mean)
                             │
              anchors A(phi,S) = {F+phi+kS < ceiling}  -> gather z[:, anchors]  (B,A,d_z)
                             │
              decoder: BaselineFutureDecoder(HorizonDecoderCore)   invoked twice, same weights
                 proj ResidualMLP d_z->256 ; core: +horizon_embedding(H,256), 4 dilated Conv1d(k3, d=1,2,4,8)+GN+GELU
                 with per-block FiLM from z, 2 horizon self-attention blocks, LayerNorm(feat+skip)
                 mean_head Linear(256->C_out), logvar_head Linear(256->C_out) -> smooth_bound(logvar_clamp)
                 (+ persistence_weight(H,C_out) * y_anchor on feature cells)
                 -> mu_base/logvar_base from z_p, mu_full/logvar_full from z_q : (B,A,H,C_out)
                             │
              kld_tensor(mu_p,logvar_p,mu_q,logvar_q) (B,T,d_z) -> te_analysis -> kld_per_t (B,T),
              source_kl_lag_map (B,T,L) = sum_m KL_head_m * alpha_m, kld_per_t_per_head (B,T,M)
```

$C_{\mathrm{out}}$ is the number of kept target channels on the CFS cells (`FeatureForecastTarget._default_decoder_out_channels`) and $16$ (`raw_per_step`) on the CRWS cells. Every stream tensor is $(B,T,\cdot)$ with $T=300$. Forecast tensors are $(B,A_{\max},H,C_{\mathrm{out}})$. The number of lags is $L=$ `max_lag` $+1$, and $M$ is the number of attention heads.

Three mechanism notes apply to the diagram:

- The persistence input is gathered from the target stream before the target gate, so it receives the current kept target values. It is a target-only decoder path outside the latent, and both branches receive the same persistence input.
- `TEAnalysisHead` keeps its name from the source. Its output is the per-head latent KL distributed over lags by the attention weights. The source-conditioned branch does not observe future target labels; "posterior" is its architectural name.
- Under `base_decode: mean` the base branch decodes $\mu^p$, while the full branch decodes a sample of its latent.

### 0.2 The four classes

| model | class (bases, in MRO order) | file | target encoder | source K/V module (`lag_kv_source` in `default.yaml`) | gates target/source (kept/declared, max shift) | $H$, $L$ | decoder out | params (trainable) |
|---|---|---|---|---|---|---|---|---|
| cfs | `SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws)` | `teb_vae/lag_attn_cfs/nets/model.py` | `CausalConvLstmEncoder` kernels (3,7,11,15,15), dilations (1,2,4,8,16), + 2-layer LSTM | `conv_stem`: `CausalConvStem` kernels (3,5,11,15,15), dilations (1,2,4,8,16), RF 387 steps | 76/80 (0) / 42/46 (0) | 10, 91 | 76 | 4,632,259 (4,615,747) |
| trf-cfs | `SeqVaeLagAttnTrfCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws)` | `teb_vae/lag_attn_transformer_cfs/nets/model.py` | `CausalConvTransformerEncoder`: 2 gated conv blocks, kernels (5,9), dilations (1,2), + 6 `CausalTransformerBlock` over the full causal prefix | `adapter`: no K/V module; keys and values are the source adapter output | 76/80 (0) / 42/46 (0) | 30, 38 | 76 | 4,159,685 (4,143,173) |
| crws | `SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws)` | `teb_vae/lag_attn_crws/nets/model.py` | as cfs | as cfs | 38/102 (6) / 17/51 (6) | 30, 91 | 16 | 4,589,907 (4,573,395) |
| trf-crws | `SeqVaeLagAttnTrfCrws(CausalRawInputs, SeqVaeLagAttnTrfRws)` | `teb_vae/lag_attn_transformer_crws/nets/model.py` | as trf-cfs | `conv_stem`: `GatedCausalConvStem`, 2 blocks, RF 21 steps | 98/102 (0) / 51/51 (0) | 30, 91 | 16 | 4,242,284 (4,225,772) |

The 16,512 frozen parameters in every model are `lag_attn.W_o` (weight $128\times128$ plus bias). They are unused because the posterior consumes per-head summaries. The trf-cfs `default.yaml` also sets `posterior_event_gate: true` (an `event_gate_head` in the posterior), `source_quality_mask: true` (a per-sample source key mask) and `forecast_ar_residual: true` (the `target_ar_logit` parameter); the other three `default.yaml` files set none of these.

### 0.3 Top-level attribute → class → source

| attribute | class | defined in | notes |
|---|---|---|---|
| `target_gate`, `source_gate` | `ChannelGate` → `ChannelDelay` | `teb_vae/lag_attn/nets/delays.py` | built if a keep_index or delays are given (`_build_channel_gate`) |
| `target_adapter`, `source_adapter` | `AvailabilityInputAdapter` | `teb_vae/lag_attn/nets/encoders.py` | the causal cells build them with delays $W'_c+d_c$ (`CausalWarmupInputs._build_adapter`) |
| `target_encoder` | `CausalConvLstmEncoder` / `CausalConvTransformerEncoder` | `lag_attn/nets/encoders.py` / `lag_attn_transformer_rws/nets/encoders.py` | |
| `source_encoder` \| `source_kv_stem` \| (none) | `CausalConvLstmEncoder` or `CausalConvStem` / `CausalConvTransformerEncoder` or `GatedCausalConvStem` | same files | at most one is built, selected by `lag_kv_source` (`encoder` \| `conv_stem` \| `adapter`) |
| `prior_head` | `FullLatentPriorHead` | `teb_vae/lag_attn_rws/nets/heads.py` | `clock_proj` Linear(128→128, no bias, zero-init) exists if `prior_availability_input` |
| `query_proj` | `nn.Linear(d_z→d_model)` | `lag_attn_rws/nets/model.py` | |
| `lag_attn` | `LagCrossAttention` | `teb_vae/lag_attn/nets/attention.py` | `lag_embeddings (L,M,d_head)`, `lag_score_bias (M,L)` (`alibi_decay` seed; flat when `alibi_slope_scale` is 0) |
| `posterior_head` | `PosteriorHead(head_structured=True)` | `teb_vae/lag_attn/nets/heads.py` | $M$ per-head fusions `ResidualMLP(160→32)`, `delta_mu_head` $M\times$Linear(32→16), `logvar_post_head` $M\times$Linear(32→16), `event_gate_head` Linear(128→1) if `posterior_event_gate` |
| `te_analysis` | `TEAnalysisHead` | `lag_attn/nets/heads.py` | parameter-free |
| `horizon_core` | `HorizonDecoderCore` | `teb_vae/lag_attn/nets/decoders.py` | shared object; also registered inside `decoder.core` (counted twice in the per-child table, once in the total) |
| `decoder` | `BaselineFutureDecoder` | `lag_attn/nets/decoders.py` | `persistence_weight (H,C_out)` on feature cells |
| `target_ar_logit` | `nn.Parameter` $(C_{\mathrm{keep}},)$ | `lag_attn_cfs/nets/causal_feature_target.py` | feature cells under `forecast_ar_residual` only |
| buffers | `future_index (T_valid,H,16)`, `horizon_weight (H,)`, `target_channel_weight`, `warm_tertile_id`, `target_warm_steps`, `novelty_tertile_id`, `target_cell_mask (H,C_keep)` (under a per-channel scored horizon), `source_block_warm_st/ph (T,)`, gate `keep_index`/`delay_steps`, adapter `availability (T,C)` and `start_indicator (T,1)`, RoPE `cos/sin_table (T,16)`, source `attn_mask (T,T)` | | all non-persistent |

### 0.4 Construction and initialisation order (`SeqVaeLagAttnRws.__init__`, mirrored by `SeqVaeLagAttnTrfRws.__init__`)

1. Mixin pre-step (causal cells): `_set_causal_inputs`. Feature cells then run `_set_channel_weights`, `_set_target_novelty` and `_set_likelihood_structure`.
2. Validate widths, heads, lags and `d_z % num_heads`; build `TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=H, warmup=warmup_period)`.
3. Build the buffers `future_index` and `horizon_weight`; the gates; the adapters (the mixin override announces $W'_c+d_c$); the target encoder; the source K/V module per `lag_kv_source`; `prior_head` (clock width `_prior_clock_dim()`, which is `d_model` on causal cells); `query_proj`; `lag_attn` (dropout 0); `posterior_head` (head-structured, with the event gate if configured); `te_analysis`; `horizon_core` (`film_per_block=True`); `decoder` (width from `_default_decoder_out_channels()`, dropout 0, persistence per `_check_persistence_target`).
4. `causal_norm` → `causalize_norms` on the target encoder and the source K/V body (conv-LSTM cells only). Freeze `lag_attn.W_o`.
5. `initialization(self)`: xavier for Linear and Conv, orthogonal for LSTM, forget bias 1. The transformer cells then run `init_depthwise_` ($\sigma=1/\sqrt{k}$ on depthwise convs).
6. Re-zero after the generic init: `_zero_init_delta_heads` (delta_mu heads zero; independent logvar head weight 0, bias $\log(5/3)$), `prior_head.zero_init_clock()`, `posterior_head.zero_init_event_gate()`, `_zero_init_film_generators`.
7. Init policy: `_reinit_horizon_embedding` (std `horizon_embed_std`) if that std is not 0.02; `_calibrate_output_heads` (mean_head ×0.02, logvar bias $\log(5/3)$, logvar weight ×0.1) and `_calibrate_prior_scale` (zero skip and final weight, bias $\log(5/3)$) if `head_init_calibration`; `_set_a_head_gain` (LayerNorm gain `a_head_gain` on the attended summary) if `a_head_gain` is not 1.
8. Mixin post-step: `_validate_causal_geometry` (stride against span, anchor floor, readout constants). Feature cells then run `_register_channel_weights` and `_register_likelihood_structure`.

---

## 1. Module trees and shape traces from `default.yaml`

### 1.1 `SeqVaeLagAttnCfs` — `teb_vae/lag_attn_cfs/configs/default.yaml`

Header (kwargs, budget, parameters, buffers, gates, geometry):

```text
kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 10, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 80, "c_u": 46, "use_up_st": true, "lstm_layers": 2, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "causal_norm": true, "anchor_stride": 5, "lag_floor": 0, "target_weight_st": 1.0, "target_weight_ph": 0.1, "horizon_weight_halflife_steps": 5.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "persistence_residual": true, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "encoder_extra_dilations": [8, 16], "encoder_extra_kernel": 15, "target_keep_index": "<76 entries, min 0 max 79>", "target_warmup_steps": "<76 entries, min 0 max 134>", "source_keep_index": "<42 entries, min 0 max 45>", "source_warmup_steps": "<42 entries, min 0 max 134>", "target_novelty_frac": "<80 entries, min 1.7040658349287696e-05 max 0.9976876378059387>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, phase operator integer_harmonic_v1, unaligned): target: fhr_st 32/36, fhr_ph 44/44; 76/80 channels, warm-up 0-134 steps; source: up_st 32/36, up_ph 10/10; 42/46 channels, warm-up 0-134 steps
params total=4,632,259 trainable=4,615,747 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                    0
  source_gate                    0
  target_adapter            86,912
  source_adapter            78,208
  target_encoder         1,344,999
  source_kv_stem           804,352
  prior_head               124,650
  query_proj                 8,320
  lag_attn                  78,572
  posterior_head           116,484
  te_analysis                    0
  horizon_core           1,844,226
  decoder                1,989,762
non-persistent buffers: [('future_index', (290, 10, 16)), ('horizon_weight', (10,)), ('warm_tertile_id', (76,)), ('target_warm_steps', (76,)), ('novelty_tertile_id', (76,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_channel_weight', (76,)), ('target_gate.keep_index', (76,)), ('target_gate.delay.delay_steps', (76,)), ('source_gate.keep_index', (42,)), ('source_gate.delay.delay_steps', (42,)), ('target_adapter.availability', (300, 76)), ('source_adapter.availability', (300, 42))]
gates (out_channels, max_delay): target=(76, 0) source=(42, 0)
decoder_out_channels=76 anchor_ceiling=290 anchor_stride=5 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=10, warmup=134)
target_forecast_shift range: None
```

Module tree:

```text
SeqVaeLagAttnCfs(
  (target_gate): ChannelGate(
    80 -> 76, max_delay=0
    (delay): ChannelDelay(num_channels=76, max_delay=0)
  )
  (source_gate): ChannelGate(
    46 -> 42, max_delay=0
    (delay): ChannelDelay(num_channels=42, max_delay=0)
  )
  (target_adapter): AvailabilityInputAdapter(
    76 -> 128, max_delay=134, availability_terms=['W_m']
    (linear): Linear(in_features=76, out_features=128, bias=True)
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
    (mask_proj): Linear(in_features=76, out_features=128, bias=False)
  )
  (source_adapter): AvailabilityInputAdapter(
    42 -> 128, max_delay=134, availability_terms=['W_m']
    (linear): Linear(in_features=42, out_features=128, bias=True)
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
    (mask_proj): Linear(in_features=42, out_features=128, bias=False)
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
    (mean_head): Linear(in_features=256, out_features=76, bias=True)
    (logvar_head): Linear(in_features=256, out_features=76, bias=True)
  )
)
```

### 1.2 `SeqVaeLagAttnTrfCfs` — `teb_vae/lag_attn_transformer_cfs/configs/default.yaml`

Header (kwargs, budget, parameters, buffers, gates, geometry):

```text
kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 80, "c_u": 46, "use_up_st": true, "dropout": 0.1, "source_dropout": 0.2, "posterior_event_gate": true, "source_quality_mask": true, "decoder_hidden": 256, "logvar_clamp": [-6.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "encoder_conv_kernels": [5, 9], "encoder_conv_dilations": [1, 2], "encoder_num_heads": 4, "encoder_d_ff": 512, "target_attention_blocks": 6, "source_attention_blocks": 3, "source_attention_window": 16, "anchor_stride": 15, "lag_floor": 0, "target_weight_st": 1.0, "target_weight_ph": 0.1, "horizon_weight_halflife_steps": 30.0, "forecast_ar_residual": true, "max_lag": 37, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "adapter", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "persistence_residual": true, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "target_keep_index": "<76 entries, min 0 max 79>", "target_warmup_steps": "<76 entries, min 0 max 134>", "source_keep_index": "<42 entries, min 0 max 45>", "source_warmup_steps": "<42 entries, min 0 max 134>", "target_novelty_frac": "<80 entries, min 0.0024794680066406727 max 1.0>", "target_scored_horizon": "<80 entries, min 5 max 30>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, phase operator integer_harmonic_v1, unaligned): target: fhr_st 32/36, fhr_ph 44/44; 76/80 channels, warm-up 0-134 steps; source: up_st 32/36, up_ph 10/10; 42/46 channels, warm-up 0-134 steps
params total=4,159,685 trainable=4,143,173 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                    0
  source_gate                    0
  target_adapter            86,912
  source_adapter            78,208
  target_encoder         1,676,928
  prior_head               124,650
  query_proj                 8,320
  lag_attn                  71,576
  posterior_head           116,613
  te_analysis                    0
  horizon_core           1,849,346
  decoder                1,996,402
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('warm_tertile_id', (76,)), ('target_warm_steps', (76,)), ('novelty_tertile_id', (76,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_channel_weight', (76,)), ('target_cell_mask', (30, 76)), ('target_gate.keep_index', (76,)), ('target_gate.delay.delay_steps', (76,)), ('source_gate.keep_index', (42,)), ('source_gate.delay.delay_steps', (42,)), ('target_adapter.availability', (300, 76)), ('source_adapter.availability', (300, 42)), ('target_encoder.attention_blocks.0.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.0.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.1.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.1.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.2.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.2.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.3.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.3.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.4.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.4.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.5.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.5.attn.rope.sin_table', (300, 16))]
gates (out_channels, max_delay): target=(76, 0) source=(42, 0)
decoder_out_channels=76 anchor_ceiling=270 anchor_stride=15 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: None
```

Module tree:

```text
SeqVaeLagAttnTrfCfs(
  (target_gate): ChannelGate(
    80 -> 76, max_delay=0
    (delay): ChannelDelay(num_channels=76, max_delay=0)
  )
  (source_gate): ChannelGate(
    46 -> 42, max_delay=0
    (delay): ChannelDelay(num_channels=42, max_delay=0)
  )
  (target_adapter): AvailabilityInputAdapter(
    76 -> 128, max_delay=134, availability_terms=['W_m']
    (linear): Linear(in_features=76, out_features=128, bias=True)
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
    (mask_proj): Linear(in_features=76, out_features=128, bias=False)
  )
  (source_adapter): AvailabilityInputAdapter(
    42 -> 128, max_delay=134, availability_terms=['W_m']
    (linear): Linear(in_features=42, out_features=128, bias=True)
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (act): GELU(approximate='none')
    (drop): Dropout(p=0.2, inplace=False)
    (res_mlp): ResidualMLP(
      (input_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (body): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (2): GELU(approximate='none')
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=128, out_features=128, bias=True)
        (5): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (6): GELU(approximate='none')
        (7): Dropout(p=0.2, inplace=False)
        (8): Linear(in_features=128, out_features=128, bias=True)
        (9): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (10): GELU(approximate='none')
        (11): Dropout(p=0.2, inplace=False)
        (12): Linear(in_features=128, out_features=128, bias=True)
      )
      (skip_proj): Identity()
    )
    (mask_proj): Linear(in_features=42, out_features=128, bias=False)
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
    (a_dropout): Dropout(p=0.2, inplace=False)
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
    (event_gate_head): Linear(in_features=128, out_features=1, bias=True)
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
    (mean_head): Linear(in_features=256, out_features=76, bias=True)
    (logvar_head): Linear(in_features=256, out_features=76, bias=True)
  )
)
```

### 1.3 `SeqVaeLagAttnCrws` — `teb_vae/lag_attn_crws/configs/default.yaml`

Header (kwargs, budget, parameters, buffers, gates, geometry):

```text
kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 102, "c_u": 51, "use_up_st": true, "lstm_layers": 2, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "causal_norm": true, "anchor_stride": 30, "lag_floor": 0, "horizon_weight_halflife_steps": 15.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "encoder_extra_dilations": [8, 16], "encoder_extra_kernel": 15, "target_keep_index": "<38 entries, min 0 max 56>", "target_warmup_steps": "<38 entries, min 0 max 1>", "source_keep_index": "<17 entries, min 0 max 16>", "source_warmup_steps": "<17 entries, min 0 max 1>", "target_align_delays": "<38 entries, min 0 max 6>", "source_align_delays": "<17 entries, min 0 max 6>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, phase operator ratio_power_v0, reference 42.2066 s): target: fhr_st 17/36, fhr_ph 21/66; 38/102 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6; source: up_st 17/36, up_ph 0/15; 17/51 channels, warm-up 0-1 steps, shift 0-6 steps, honest by 0-6
params total=4,589,907 trainable=4,573,395 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                    0
  source_gate                    0
  target_adapter            77,312
  source_adapter            71,936
  target_encoder         1,344,999
  source_kv_stem           804,352
  prior_head               124,650
  query_proj                 8,320
  lag_attn                  78,572
  posterior_head           116,484
  te_analysis                    0
  horizon_core           1,849,346
  decoder                1,963,282
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_gate.keep_index', (38,)), ('target_gate.delay.delay_steps', (38,)), ('source_gate.keep_index', (17,)), ('source_gate.delay.delay_steps', (17,)), ('target_adapter.availability', (300, 38)), ('target_adapter.start_indicator', (300, 1)), ('source_adapter.availability', (300, 17)), ('source_adapter.start_indicator', (300, 1))]
gates (out_channels, max_delay): target=(38, 6) source=(17, 6)
decoder_out_channels=16 anchor_ceiling=270 anchor_stride=30 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: None
```

Module tree: the `SeqVaeLagAttnCfs` tree above, except for these lines (unified diff, `SeqVaeLagAttnCfs` → `SeqVaeLagAttnCrws`):

```diff
@@ -1,13 +1,13 @@
-SeqVaeLagAttnCfs(
+SeqVaeLagAttnCrws(
   (target_gate): ChannelGate(
-    80 -> 76, max_delay=0
-    (delay): ChannelDelay(num_channels=76, max_delay=0)
+    102 -> 38, max_delay=6
+    (delay): ChannelDelay(num_channels=38, max_delay=6)
   )
   (source_gate): ChannelGate(
-    46 -> 42, max_delay=0
-    (delay): ChannelDelay(num_channels=42, max_delay=0)
+    51 -> 17, max_delay=6
+    (delay): ChannelDelay(num_channels=17, max_delay=6)
   )
   (target_adapter): AvailabilityInputAdapter(
-    76 -> 128, max_delay=134, availability_terms=['W_m']
-    (linear): Linear(in_features=76, out_features=128, bias=True)
+    38 -> 128, max_delay=6, availability_terms=['W_m', 'e_start']
+    (linear): Linear(in_features=38, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
@@ -34,7 +34,7 @@
     )
-    (mask_proj): Linear(in_features=76, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=38, out_features=128, bias=False)
   )
   (source_adapter): AvailabilityInputAdapter(
-    42 -> 128, max_delay=134, availability_terms=['W_m']
-    (linear): Linear(in_features=42, out_features=128, bias=True)
+    17 -> 128, max_delay=6, availability_terms=['W_m', 'e_start']
+    (linear): Linear(in_features=17, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
@@ -61,3 +61,3 @@
     )
-    (mask_proj): Linear(in_features=42, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=17, out_features=128, bias=False)
   )
@@ -357,4 +357,4 @@
     )
-    (mean_head): Linear(in_features=256, out_features=76, bias=True)
-    (logvar_head): Linear(in_features=256, out_features=76, bias=True)
+    (mean_head): Linear(in_features=256, out_features=16, bias=True)
+    (logvar_head): Linear(in_features=256, out_features=16, bias=True)
   )
```

### 1.4 `SeqVaeLagAttnTrfCrws` — `teb_vae/lag_attn_transformer_crws/configs/default.yaml`

Header (kwargs, budget, parameters, buffers, gates, geometry):

```text
kwargs (from default.yaml + resolved budget): {"d_model": 128, "d_z": 64, "horizon": 30, "raw_per_step": 16, "warmup_period": 134, "sequence_length": 300, "c_y": 102, "c_u": 51, "use_up_st": true, "dropout": 0.1, "decoder_hidden": 256, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0, "delta_mu_scale": 3.0, "delta_logvar_scale": 2.0, "coverage_floor": 0.9, "base_decode": "mean", "posterior_logvar_mode": "independent", "encoder_conv_kernels": [5, 9], "encoder_conv_dilations": [1, 2], "encoder_num_heads": 4, "encoder_d_ff": 512, "target_attention_blocks": 6, "source_attention_blocks": 3, "source_attention_window": 16, "anchor_stride": 30, "lag_floor": 0, "horizon_weight_halflife_steps": 15.0, "max_lag": 90, "num_heads": 4, "d_head": 32, "use_entmax": true, "attention_grad_checkpoint": false, "lag_kv_source": "conv_stem", "lag_bias_init": "alibi_decay", "alibi_slope_scale": 0.0, "query_uses_logvar": false, "prior_availability_input": true, "horizon_depth": 4, "horizon_kernel": 3, "horizon_film": true, "horizon_attention_blocks": 2, "horizon_embed_std": 0.8, "head_init_calibration": true, "a_head_gain": 2.0, "target_keep_index": "<98 entries, min 0 max 101>", "target_warmup_steps": "<98 entries, min 0 max 134>", "source_keep_index": "<51 entries, min 0 max 50>", "source_warmup_steps": "<51 entries, min 0 max 278>"}
budget summary: causal warm-up budget 134 steps (quantile 0.95, trim_minutes 1.0, leg alignment envelope, phase operator ratio_power_v0, unaligned): target: fhr_st 32/36, fhr_ph 66/66; 98/102 channels, warm-up 0-134 steps; source: up_st 36/36, up_ph 15/15; 51/51 channels, warm-up 0-278 steps
params total=4,242,284 trainable=4,225,772 (frozen = lag_attn.W_o)
per top-level child (params):
  target_gate                    0
  source_gate                    0
  target_adapter            92,544
  source_adapter            80,512
  target_encoder         1,676,928
  source_kv_stem           100,992
  prior_head               124,650
  query_proj                 8,320
  lag_attn                  78,572
  posterior_head           116,484
  te_analysis                    0
  horizon_core           1,849,346
  decoder                1,963,282
non-persistent buffers: [('future_index', (270, 30, 16)), ('horizon_weight', (30,)), ('source_block_warm_st', (300,)), ('source_block_warm_ph', (300,)), ('target_gate.keep_index', (98,)), ('target_gate.delay.delay_steps', (98,)), ('source_gate.keep_index', (51,)), ('source_gate.delay.delay_steps', (51,)), ('target_adapter.availability', (300, 98)), ('source_adapter.availability', (300, 51)), ('target_encoder.attention_blocks.0.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.0.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.1.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.1.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.2.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.2.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.3.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.3.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.4.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.4.attn.rope.sin_table', (300, 16)), ('target_encoder.attention_blocks.5.attn.rope.cos_table', (300, 16)), ('target_encoder.attention_blocks.5.attn.rope.sin_table', (300, 16))]
gates (out_channels, max_delay): target=(98, 0) source=(51, 0)
decoder_out_channels=16 anchor_ceiling=270 anchor_stride=30 geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=134)
target_forecast_shift range: None
```

Module tree: the `SeqVaeLagAttnTrfCfs` tree above, except for these lines (unified diff, `SeqVaeLagAttnTrfCfs` → `SeqVaeLagAttnTrfCrws`):

```diff
@@ -1,13 +1,13 @@
-SeqVaeLagAttnTrfCfs(
+SeqVaeLagAttnTrfCrws(
   (target_gate): ChannelGate(
-    80 -> 76, max_delay=0
-    (delay): ChannelDelay(num_channels=76, max_delay=0)
+    102 -> 98, max_delay=0
+    (delay): ChannelDelay(num_channels=98, max_delay=0)
   )
   (source_gate): ChannelGate(
-    46 -> 42, max_delay=0
-    (delay): ChannelDelay(num_channels=42, max_delay=0)
+    51 -> 51, max_delay=0
+    (delay): ChannelDelay(num_channels=51, max_delay=0)
   )
   (target_adapter): AvailabilityInputAdapter(
-    76 -> 128, max_delay=134, availability_terms=['W_m']
-    (linear): Linear(in_features=76, out_features=128, bias=True)
+    98 -> 128, max_delay=134, availability_terms=['W_m']
+    (linear): Linear(in_features=98, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
@@ -34,10 +34,10 @@
     )
-    (mask_proj): Linear(in_features=76, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=98, out_features=128, bias=False)
   )
   (source_adapter): AvailabilityInputAdapter(
-    42 -> 128, max_delay=134, availability_terms=['W_m']
-    (linear): Linear(in_features=42, out_features=128, bias=True)
+    51 -> 128, max_delay=278, availability_terms=['W_m']
+    (linear): Linear(in_features=51, out_features=128, bias=True)
     (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
     (act): GELU(approximate='none')
-    (drop): Dropout(p=0.2, inplace=False)
+    (drop): Dropout(p=0.1, inplace=False)
     (res_mlp): ResidualMLP(
@@ -48,3 +48,3 @@
         (2): GELU(approximate='none')
-        (3): Dropout(p=0.2, inplace=False)
+        (3): Dropout(p=0.1, inplace=False)
         (4): Linear(in_features=128, out_features=128, bias=True)
@@ -52,3 +52,3 @@
         (6): GELU(approximate='none')
-        (7): Dropout(p=0.2, inplace=False)
+        (7): Dropout(p=0.1, inplace=False)
         (8): Linear(in_features=128, out_features=128, bias=True)
@@ -56,3 +56,3 @@
         (10): GELU(approximate='none')
-        (11): Dropout(p=0.2, inplace=False)
+        (11): Dropout(p=0.1, inplace=False)
         (12): Linear(in_features=128, out_features=128, bias=True)
@@ -61,3 +61,3 @@
     )
-    (mask_proj): Linear(in_features=42, out_features=128, bias=False)
+    (mask_proj): Linear(in_features=51, out_features=128, bias=False)
   )
@@ -112,2 +112,32 @@
         (ffn_scale): LayerScale(128, init=0.01)
+      )
+    )
+    (output_norm): RMSNorm(128, eps=1e-05)
+  )
+  (source_kv_stem): GatedCausalConvStem(
+    d_model=128, conv_blocks=2, receptive_field=21 steps
+    (conv_blocks): ModuleList(
+      (0): GatedCausalConvBlock(
+        (norm_in): RMSNorm(128, eps=1e-05)
+        (proj_in): Linear(in_features=128, out_features=256, bias=False)
+        (conv): CausalDepthwiseConv1d(
+          128, kernel_size=5, dilation=1, left_padding=4
+          (conv): Conv1d(128, 128, kernel_size=(5,), stride=(1,), groups=128, bias=False)
+        )
+        (norm_conv): RMSNorm(128, eps=1e-05)
+        (proj_out): Linear(in_features=128, out_features=128, bias=False)
+        (dropout): Dropout(p=0.1, inplace=False)
+        (layer_scale): LayerScale(128, init=0.01)
+      )
+      (1): GatedCausalConvBlock(
+        (norm_in): RMSNorm(128, eps=1e-05)
+        (proj_in): Linear(in_features=128, out_features=256, bias=False)
+        (conv): CausalDepthwiseConv1d(
+          128, kernel_size=9, dilation=2, left_padding=16
+          (conv): Conv1d(128, 128, kernel_size=(9,), stride=(1,), dilation=(2,), groups=128, bias=False)
+        )
+        (norm_conv): RMSNorm(128, eps=1e-05)
+        (proj_out): Linear(in_features=128, out_features=128, bias=False)
+        (dropout): Dropout(p=0.1, inplace=False)
+        (layer_scale): LayerScale(128, init=0.01)
       )
@@ -180,3 +210,3 @@
     (h_y_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
-    (a_dropout): Dropout(p=0.2, inplace=False)
+    (a_dropout): Dropout(p=0.0, inplace=False)
     (a_head_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
@@ -207,3 +237,2 @@
     )
-    (event_gate_head): Linear(in_features=128, out_features=1, bias=True)
   )
@@ -299,4 +328,4 @@
     )
-    (mean_head): Linear(in_features=256, out_features=76, bias=True)
-    (logvar_head): Linear(in_features=256, out_features=76, bias=True)
+    (mean_head): Linear(in_features=256, out_features=16, bias=True)
+    (logvar_head): Linear(in_features=256, out_features=16, bias=True)
   )
```

### 1.5 Forward shape traces (random inputs, `eval()` mode, forward hooks on top-level children)

The source-side hook lines at batch 1 are the prior clock: `_prior_clock` re-runs `source_gate → source_adapter →` K/V module on a zero stream of batch 1. Under `lag_kv_source: adapter` no K/V module runs. Under `source_quality_mask`, the extra `source_gate` line at batch 2 is `source_key_mask`, which passes the source-validity mask through the gate. The decoder appears twice (prior latent, then posterior latent) with the same weights.

#### `SeqVaeLagAttnCfs`: forward at B=2, T=300, anchor_stride=5, anchor_phase=0

```text
  target_gate          in=[(2, 300, 80)] out=(2, 300, 76)
  source_gate          in=[(2, 300, 46)] out=(2, 300, 42)
  target_adapter       in=[(2, 300, 76)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 42)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 46)] out=(1, 300, 42)
  source_adapter       in=[(1, 300, 42)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 32, 64)] out=[(2, 32, 10, 76), (2, 32, 10, 76)]
  decoder              in=[(2, 32, 64)] out=[(2, 32, 10, 76), (2, 32, 10, 76)]
  te_analysis          in=[(2, 300, 64), (2, 300, 4, 91)] out=[(2, 300), (2, 300, 91), (2, 300, 4)]
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
  mu_base                  (2, 32, 10, 76) float32
  logvar_base              (2, 32, 10, 76) float32
  mu_full                  (2, 32, 10, 76) float32
  logvar_full              (2, 32, 10, 76) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 32) int64
  anchor_valid             (2, 32) bool
  persistence              (2, 32, 76) float32
```

#### `SeqVaeLagAttnCfs` (dense validation/test geometry): forward at B=2, T=300, anchor_stride=1, anchor_phase=0

```text
  target_gate          in=[(2, 300, 80)] out=(2, 300, 76)
  source_gate          in=[(2, 300, 46)] out=(2, 300, 42)
  target_adapter       in=[(2, 300, 76)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 42)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 46)] out=(1, 300, 42)
  source_adapter       in=[(1, 300, 42)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 156, 64)] out=[(2, 156, 10, 76), (2, 156, 10, 76)]
  decoder              in=[(2, 156, 64)] out=[(2, 156, 10, 76), (2, 156, 10, 76)]
  te_analysis          in=[(2, 300, 64), (2, 300, 4, 91)] out=[(2, 300), (2, 300, 91), (2, 300, 4)]
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
  mu_base                  (2, 156, 10, 76) float32
  logvar_base              (2, 156, 10, 76) float32
  mu_full                  (2, 156, 10, 76) float32
  logvar_full              (2, 156, 10, 76) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 91) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 156) int64
  anchor_valid             (2, 156) bool
  persistence              (2, 156, 76) float32
```

#### `SeqVaeLagAttnTrfCfs`: forward at B=2, T=300, anchor_stride=15, anchor_phase=0

```text
  target_gate          in=[(2, 300, 80)] out=(2, 300, 76)
  source_gate          in=[(2, 300, 46)] out=(2, 300, 42)
  target_adapter       in=[(2, 300, 76)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 42)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 46)] out=(1, 300, 42)
  source_adapter       in=[(1, 300, 42)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  source_gate          in=[(2, 300, 46)] out=(2, 300, 42)
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (2, 300, 38)] out=[(2, 300, 128), (2, 300, 4, 38), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 10, 64)] out=[(2, 10, 30, 76), (2, 10, 30, 76)]
  decoder              in=[(2, 10, 64)] out=[(2, 10, 30, 76), (2, 10, 30, 76)]
  te_analysis          in=[(2, 300, 64), (2, 300, 4, 38)] out=[(2, 300), (2, 300, 38), (2, 300, 4)]
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
  attn_weights             (2, 300, 4, 38) float32
  mu_base                  (2, 10, 30, 76) float32
  logvar_base              (2, 10, 30, 76) float32
  mu_full                  (2, 10, 30, 76) float32
  logvar_full              (2, 10, 30, 76) float32
  kld_per_t                (2, 300) float32
  kld_per_t_per_head       (2, 300, 4) float32
  source_kl_lag_map        (2, 300, 38) float32
  mu_prior_sat_frac        scalar float32
  delta_mu_sat_frac        scalar float32
  anchor_index             (2, 10) int64
  anchor_valid             (2, 10) bool
  persistence              (2, 10, 76) float32
  event_gate               (2, 300, 1) float32
  source_key_mask          (2, 300, 38) bool
  source_open              (2, 300) bool
```

#### `SeqVaeLagAttnCrws`: forward at B=2, T=300, anchor_stride=30, anchor_phase=0

```text
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
  te_analysis          in=[(2, 300, 64), (2, 300, 4, 91)] out=[(2, 300), (2, 300, 91), (2, 300, 4)]
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

#### `SeqVaeLagAttnTrfCrws`: forward at B=2, T=300, anchor_stride=30, anchor_phase=0

```text
  target_gate          in=[(2, 300, 102)] out=(2, 300, 98)
  source_gate          in=[(2, 300, 51)] out=(2, 300, 51)
  target_adapter       in=[(2, 300, 98)] out=(2, 300, 128)
  target_encoder       in=[(2, 300, 128)] out=(2, 300, 128)
  source_adapter       in=[(2, 300, 51)] out=(2, 300, 128)
  source_kv_stem       in=[(2, 300, 128)] out=(2, 300, 128)
  source_gate          in=[(1, 300, 51)] out=(1, 300, 51)
  source_adapter       in=[(1, 300, 51)] out=(1, 300, 128)
  source_kv_stem       in=[(1, 300, 128)] out=(1, 300, 128)
  prior_head           in=[(2, 300, 128)] out=[(2, 300, 64), (2, 300, 64), (2, 300, 64)]
  query_proj           in=[(2, 300, 64)] out=(2, 300, 128)
  lag_attn             in=[(2, 300, 128), (2, 300, 128), (300, 91)] out=[(2, 300, 128), (2, 300, 4, 91), (2, 300, 4, 32)]
  posterior_head       in=[(2, 300, 128), (2, 300, 4, 32), (2, 300, 64), (2, 300, 64)] out=[(2, 300, 64), (2, 300, 64)]
  decoder              in=[(2, 5, 64)] out=[(2, 5, 30, 16), (2, 5, 30, 16)]
  decoder              in=[(2, 5, 64)] out=[(2, 5, 30, 16), (2, 5, 30, 16)]
  te_analysis          in=[(2, 300, 64), (2, 300, 4, 91)] out=[(2, 300), (2, 300, 91), (2, 300, 4)]
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

`ChannelGate.forward(x (B,T,C_declared)) = ChannelDelay(index_select(x, keep_index))`; `ChannelDelay` reads channel $c$ at step $t-\delta_c$ (clamped) and zeroes steps $t<\delta_c$. In the causal cells `delays` are the alignment shifts $d_c$ (companion §1.4), which are zeros when no alignment reference is set.

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

`AvailabilityInputAdapter`: `Linear(in→128)` on the masked input + `mask_proj(availability−1)` + `start_embed` where no channel is available, then LayerNorm → GELU → Dropout → `ResidualMLP(128→(128,128,128,128), final_activation=False)`. The repr lists the availability terms that the adapter built (`availability_terms`). `CausalConvLstmEncoder`: `front_mlp` → [conv stack ‖ LSTM] → `fusion ResidualMLP(256→…→128)` → LayerNorm. `CausalConvStem`: conv stack + LayerNorm(out + x), receptive field $1+\sum(k-1)r$, which is 387 steps for kernels (3,5,11,15,15) and dilations (1,2,4,8,16).

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

Keys and values are unfolded into a $(B,T,M,d,L)$ window of the previous $L$ steps ($L=$ `max_lag` $+1$). Window index $j$ corresponds to lag $L-1-j$ (hence the `.flip` calls). Scores = content + Shaw relative key bias (`lag_embeddings`), scaled by $1/\sqrt{d}$, plus the unscaled `lag_score_bias` (ALiBi-shaped, learnable; all zeros at `alibi_slope_scale=0`). `entmax15` or softmax over lags; masked lags get $-\infty$; all-masked rows give 0. The mask is $(T,L)$, or a per-sample $(B,T,L)$ key mask under `source_quality_mask`. Returns `W_o(head_out)` (frozen, unused), `alpha (B,T,M,L)` in lag order, `head_out (B,T,M,d_head)`.

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
  202|         batch, seq_len = h_y.shape[0], h_y.shape[1]
  203|         num_lags = self.L
  204|         heads = self.num_heads
  205|         width = self.d_head
  206|
  208|         q = self.W_q(self.q_norm(h_y)).view(batch, seq_len, heads, width)
  209|         h_u_normed = self.kv_norm(h_u)
  210|         k = self.W_k(h_u_normed).view(batch, seq_len, heads, width)
  211|         v = self.W_v(h_u_normed).view(batch, seq_len, heads, width)
  212|
  216|         k_padded = F.pad(k, (0, 0, 0, 0, num_lags - 1, 0))  # (B, T+L-1, Mh, d)
  217|         v_padded = F.pad(v, (0, 0, 0, 0, num_lags - 1, 0))
  218|         k_window = k_padded.unfold(1, num_lags, 1)          # (B, T, Mh, d, L) view
  219|         v_window = v_padded.unfold(1, num_lags, 1)          # (B, T, Mh, d, L) view
  220|
  222|         scores = torch.einsum("btmd,btmdj->btmj", q, k_window)
  223|         scores = scores + torch.einsum("btmd,jmd->btmj", q, self.lag_embeddings.flip(0))
  224|         scores = scores * self.scale
  225|         if self.lag_score_bias is not None:
  226|             scores = scores + self.lag_score_bias.flip(-1)[None, None, :, :]
  227|
  230|         mask_window = m_lag.flip(-1).to(torch.bool)
  231|         if mask_window.dim() == 2:
  232|             mask_window = mask_window[None]
  233|         mask_window = mask_window[:, :, None, :]  # (1 or B, T, 1, L)
  237|         any_valid = mask_window.any(dim=-1, keepdim=True)
  238|         scores = scores.masked_fill(~mask_window, float("-inf")).masked_fill(~any_valid, 0.0)
  239|
  241|         alpha_window = (
  242|             cast(torch.Tensor, entmax15(scores, dim=-1))
  243|             if self.use_entmax
  244|             else F.softmax(scores, dim=-1)
  245|         )
  248|         alpha_window = alpha_window * any_valid
  249|         alpha_window = self.attn_dropout(alpha_window)
  250|
  251|         head_out = torch.einsum("btmj,btmdj->btmd", alpha_window, v_window)
  252|         out = self.W_o(head_out.reshape(batch, seq_len, heads * width))
  253|
  254|         alpha = alpha_window.flip(-1)  # back to lag order
  257|         return out, alpha, head_out
  258|
  259|     def forward(
  260|         self,
  261|         h_y: torch.Tensor,
  262|         h_u: torch.Tensor,
  263|         m_lag: Optional[torch.Tensor] = None,
  264|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  279|         if m_lag is None:
  280|             m_lag = self.build_lag_mask(h_y.shape[1], device=h_y.device)
  281|         if self.grad_checkpoint and self.training:
  282|             return cast(
  283|                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  284|                 checkpoint(self._attend, h_y, h_u, m_lag, use_reentrant=False),
  285|             )
  286|         return self._attend(h_y, h_u, m_lag)
```

## 6. Latent heads

### 6.1 `teb_vae/lag_attn/nets/heads.py` — `PriorHead` (ancestor only, unused by these four), `PosteriorHead`, `TEAnalysisHead`

Head-structured posterior (`head_structured=True` in every model here): for head $m$, `fusion[m](cat[LN(h_y), a_dropout(LN_gain(a[:,:,m,:]))]) → (B,T,32)`, `delta_mu_head[m] → (B,T,16)`; concatenated over heads → `(B,T,64)`; $\mu^q=\mu^p+3\tanh(\cdot/3)$. `posterior_logvar_mode='independent'`: `logvar_post_head[m]` on the same fused features → `smooth_bound`. Under `event_gate=True` the head builds `event_gate_head`, and `event_gate_values` returns $g_t=\sigma(w^\top\tilde a_t+b)$ $(B,T,1)$ from the normalised attended summary. The gate multiplies both bounded residuals, so $q=p$ where $g_t=0$. `TEAnalysisHead`: `kld_per_head = kld_btd.view(B,T,M,16).sum(-1)`, `te_lag_map = einsum(kld_per_head, alpha)`.

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
   43| EVENT_GATE_BIAS_INIT = 2.0
   44|
   46| class PriorHead(nn.Module):
   64|
   65|     def __init__(
   66|         self,
   67|         d_model: int = 128,
   68|         d_z: int = 24,
   69|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
   70|         dropout: float = 0.1,
   71|         mu_scale: float = 5.0,
   72|     ) -> None:
   86|         super().__init__()
   87|         if mu_scale <= 0.0:
   88|             raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
   89|         self.logvar_clamp = logvar_clamp
   90|         self.mu_scale = float(mu_scale)
   91|
   93|         self.mu_input_norm = nn.LayerNorm(d_model)
   94|         self.logvar_input_norm = nn.LayerNorm(d_model)
   95|         self.dec_input_norm = nn.LayerNorm(d_model)
   96|
   97|         self.mu_prior_head = ResidualMLP(
   98|             input_dim=d_model,
   99|             hidden_dims=geometric_schedule(d_model, d_z, 4),
  100|             final_activation=False,
  101|             use_skip_connection=True,
  102|             use_input_layer_norm=True,
  103|             activation=nn.GELU,
  104|             dropout=dropout,
  105|         )
  106|         self.logvar_prior_head = ResidualMLP(
  107|             input_dim=d_model,
  108|             hidden_dims=geometric_schedule(d_model, d_z, 4),
  109|             final_activation=False,
  110|             use_skip_connection=True,
  111|             use_input_layer_norm=True,
  112|             activation=nn.GELU,
  113|             dropout=dropout,
  114|         )
  115|         self.decoder_state_head = ResidualMLP(
  116|             input_dim=d_model,
  117|             hidden_dims=geometric_schedule(d_model, d_model, 3),
  118|             final_activation=True,
  119|             use_skip_connection=True,
  120|             use_input_layer_norm=True,
  121|             activation=nn.GELU,
  122|             dropout=dropout,
  123|         )
  124|
  125|     def forward(
  126|         self, h_y: torch.Tensor
  127|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  136|         raw_mu = self.mu_prior_head(self.mu_input_norm(h_y))
  137|         mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)
  138|
  139|         raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(h_y))
  140|         logvar_prior = smooth_bound(raw_logvar_prior, *self.logvar_clamp)
  141|
  142|         decoder_state = self.decoder_state_head(self.dec_input_norm(h_y))
  143|         return mu_prior, logvar_prior, decoder_state, raw_logvar_prior
  144|
  146| class PosteriorHead(nn.Module):
  167|
  168|     def __init__(
  169|         self,
  170|         d_model: int = 128,
  171|         d_z: int = 24,
  172|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  173|         dropout: float = 0.1,
  174|         delta_mu_scale: float = 3.0,
  175|         head_structured: bool = False,
  176|         num_heads: int = 4,
  177|         d_head: int = 32,
  178|         *,
  179|         delta_logvar_scale: float = 2.0,
  180|         posterior_logvar_mode: str = "residual",
  181|         source_dropout: float = 0.0,
  182|         event_gate: bool = False,
  183|     ) -> None:
  242|         super().__init__()
  243|         if delta_mu_scale <= 0.0:
  244|             raise ValueError(f"delta_mu_scale must be > 0, got {delta_mu_scale}")
  245|         if delta_logvar_scale <= 0.0:
  246|             raise ValueError(f"delta_logvar_scale must be > 0, got {delta_logvar_scale}")
  247|
  248|         self.logvar_clamp = logvar_clamp
  249|         self.delta_mu_scale = float(delta_mu_scale)
  250|         self.delta_logvar_scale = float(delta_logvar_scale)
  251|         self.head_structured = bool(head_structured)
  252|         self.num_heads = int(num_heads)
  253|         self.posterior_logvar_mode = validate_choice(
  254|             posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
  255|         )
  256|
  258|         self.h_y_norm = nn.LayerNorm(d_model)
  268|         self.a_dropout = nn.Dropout(float(source_dropout))
  269|
  270|         if self.head_structured:
  271|             if d_z % num_heads != 0:
  272|                 raise ValueError(
  273|                     f"head_structured posterior needs d_z % num_heads == 0, "
  274|                     f"got d_z={d_z}, num_heads={num_heads}"
  275|                 )
  276|             self.group = d_z // num_heads
  277|             self.d_head = int(d_head)
  278|             self.a_head_norm = nn.LayerNorm(d_head)
  279|             fuse_in = d_model + d_head
  280|             fuse_out = max(2 * self.group, 16)
  282|             self.fusion = nn.ModuleList(
  283|                 [
  284|                     ResidualMLP(
  285|                         input_dim=fuse_in,
  286|                         hidden_dims=geometric_schedule(fuse_in, fuse_out, 2),
  287|                         final_activation=True,
  288|                         use_skip_connection=True,
  289|                         use_input_layer_norm=True,
  290|                         activation=nn.GELU,
  291|                         dropout=dropout,
  292|                     )
  293|                     for _ in range(num_heads)
  294|                 ]
  295|             )
  296|             self.delta_mu_head = nn.ModuleList(
  297|                 [nn.Linear(fuse_out, self.group) for _ in range(num_heads)]
  298|             )
  303|             logvar_head: nn.Module = nn.ModuleList(
  304|                 [nn.Linear(fuse_out, self.group) for _ in range(num_heads)]
  305|             )
  306|         else:
  308|             self.a_norm = nn.LayerNorm(d_model)
  309|             fused_in = 2 * d_model
  310|             self.fusion = ResidualMLP(
  311|                 input_dim=fused_in,
  312|                 hidden_dims=geometric_schedule(fused_in, d_model, 3),
  313|                 final_activation=True,
  314|                 use_skip_connection=True,
  315|                 use_input_layer_norm=True,
  316|                 activation=nn.GELU,
  317|                 dropout=dropout,
  318|             )
  319|             self.delta_mu_head = nn.Linear(d_model, d_z)
  320|             logvar_head = nn.Linear(d_model, d_z)
  321|
  325|         self.delta_logvar_head = (
  326|             logvar_head if self.posterior_logvar_mode == "residual" else None
  327|         )
  328|         self.logvar_post_head = (
  329|             logvar_head if self.posterior_logvar_mode == "independent" else None
  330|         )
  331|
  335|         self.event_gate_head: Optional[nn.Linear] = None
  336|         if event_gate:
  337|             gate_in = self.num_heads * self.d_head if self.head_structured else d_model
  338|             self.event_gate_head = nn.Linear(gate_in, 1)
  339|             self.zero_init_event_gate()
  340|
  341|     def zero_init_event_gate(self) -> None:
  348|         if self.event_gate_head is None:
  349|             return
  350|         nn.init.zeros_(self.event_gate_head.weight)
  351|         nn.init.constant_(self.event_gate_head.bias, EVENT_GATE_BIAS_INIT)
  352|
  353|     def event_gate_values(self, a: torch.Tensor) -> Optional[torch.Tensor]:
  367|         if self.event_gate_head is None:
  368|             return None
  369|         if self.head_structured:
  370|             summary = torch.cat(
  371|                 [self.a_head_norm(a[:, :, index, :]) for index in range(self.num_heads)], dim=-1
  372|             )
  373|         else:
  374|             summary = self.a_norm(a)
  375|         return torch.sigmoid(self.event_gate_head(summary))
  376|
  377|     def forward(
  378|         self,
  379|         h_y: torch.Tensor,
  380|         a: torch.Tensor,
  381|         mu_prior: torch.Tensor,
  382|         raw_logvar_prior: Optional[torch.Tensor] = None,
  383|         gate: Optional[torch.Tensor] = None,
  384|     ) -> Tuple[torch.Tensor, torch.Tensor]:
  407|         if raw_logvar_prior is None:
  408|             raise ValueError(
  409|                 "the posterior log-variance is built against the prior's pre-bound raw "
  410|                 "log-variance, so raw_logvar_prior is required; call via the model's forward "
  411|                 "or encode_only, which thread it through"
  412|             )
  413|
  414|         fused_heads: List[torch.Tensor] = []
  415|         fused_flat: Optional[torch.Tensor] = None
  416|         if self.head_structured:
  418|             fusion = cast(nn.ModuleList, self.fusion)
  419|             delta_mu_head = cast(nn.ModuleList, self.delta_mu_head)
  420|
  421|             h_y_normed = self.h_y_norm(h_y)
  422|             raw_deltas = []
  423|             for index in range(self.num_heads):
  424|                 a_head = self.a_dropout(self.a_head_norm(a[:, :, index, :]))
  425|                 fused_head = fusion[index](torch.cat([h_y_normed, a_head], dim=-1))
  426|                 fused_heads.append(fused_head)
  427|                 raw_deltas.append(delta_mu_head[index](fused_head))
  428|             raw_delta = torch.cat(raw_deltas, dim=-1)
  429|         else:
  430|             fused_flat = self.fusion(
  431|                 torch.cat([self.h_y_norm(h_y), self.a_dropout(self.a_norm(a))], dim=-1)
  432|             )
  433|             raw_delta = self.delta_mu_head(fused_flat)
  434|
  437|         delta_mu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
  438|         if gate is not None:
  440|             delta_mu = delta_mu * gate
  441|         mu_post = mu_prior + delta_mu
  442|
  443|         raw_logvar_post = self._run_logvar_head(fused_heads, fused_flat)
  444|         if self.delta_logvar_head is not None:
  445|             delta_logvar = (
  446|                 self.delta_logvar_scale * torch.tanh(raw_logvar_post / self.delta_logvar_scale)
  447|             )
  448|             if gate is not None:
  449|                 delta_logvar = delta_logvar * gate
  452|             logvar_post = smooth_bound(raw_logvar_prior + delta_logvar, *self.logvar_clamp)
  453|         else:
  454|             if gate is not None:
  458|                 raw_logvar_post = gate * raw_logvar_post + (1.0 - gate) * raw_logvar_prior.detach()
  462|             logvar_post = smooth_bound(raw_logvar_post, *self.logvar_clamp)
  463|
  464|         return mu_post, logvar_post
  465|
  466|     def _run_logvar_head(
  467|         self, fused_heads: List[torch.Tensor], fused_flat: Optional[torch.Tensor]
  468|     ) -> torch.Tensor:
  483|         head = self.delta_logvar_head if self.logvar_post_head is None else self.logvar_post_head
  484|         if self.head_structured:
  485|             modules = cast(nn.ModuleList, head)
  486|             return torch.cat(
  487|                 [modules[index](fused_heads[index]) for index in range(self.num_heads)], dim=-1
  488|             )
  489|         return cast(nn.Linear, head)(fused_flat)
  490|
  492| class TEAnalysisHead(nn.Module):
  506|
  507|     def forward(
  508|         self,
  509|         kld_btd: torch.Tensor,
  510|         attn_weights: torch.Tensor,
  511|         head_structured: bool = False,
  512|     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  525|         batch, seq_len, d_z = kld_btd.shape
  526|         num_heads = attn_weights.shape[2]
  527|         kld_per_t = kld_btd.sum(dim=-1)
  528|
  529|         if d_z % num_heads == 0:
  530|             group = d_z // num_heads
  531|             kld_per_head = kld_btd.view(batch, seq_len, num_heads, group).sum(dim=-1)
  532|         else:  # pragma: no cover - d_z is divisible by num_heads in practice
  533|             kld_per_head = kld_per_t.unsqueeze(-1).expand(batch, seq_len, num_heads) / num_heads
  534|
  535|         if head_structured and d_z % num_heads == 0:
  536|             te_lag_map = torch.einsum("btm,btml->btl", kld_per_head, attn_weights)
  537|         else:
  538|             mean_alpha = attn_weights.mean(dim=-2)
  539|             te_lag_map = kld_per_t.unsqueeze(-1) * mean_alpha
  540|
  541|         return kld_per_t, te_lag_map, kld_per_head
```

### 6.2 `teb_vae/lag_attn_rws/nets/heads.py` — `FullLatentPriorHead`

Two `ResidualMLP(128→(111,97,84,74,64), no final act)` on LayerNorm'd $h_y$ (+ `clock_proj(LN(clock))` added to $h_y$ when built); $\mu^p=5\tanh(\cdot/5)$, $\ell^p=\mathrm{smooth\_bound}(\cdot,\ell_{\min},\ell_{\max})$ with $(\ell_{\min},\ell_{\max})=$ `logvar_clamp`; also returns the raw pre-bound log-variance.

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

`BaselineFutureDecoder.forward(z (B,A,64), persistence (B,A,C)|None)`: `proj ResidualMLP(64→(91,128,181,256), GELU)`; `core.decode`: broadcast over $H$ + `horizon_embedding (H,256)`; fold to `(B·A, 256, H)`; `_HorizonRefine` 4 blocks `Conv1d(256,256,k=3, dil 1,2,4,8, symmetric pad)` → GroupNorm(8) → GELU → per-block FiLM $(1+\gamma)y+\beta$ from `film[i](z_proj)` → residual; 2 × `_HorizonSelfAttention` over the $H$ horizon tokens (pre-LN, bias-free q/k/v/out, SDPA, residual gain init 0.01); `out_norm(feat + skip)`. Heads: `mean_head Linear(256→C)` (+ `persistence_weight (H,C) * y_anchor`), `logvar_head → smooth_bound(logvar_clamp)`. `ResidualFutureDecoder` is the ancestor's second decoder; none of the four models build it.

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
  153|         posterior_event_gate: bool = False,
  154|         lag_kv_source: str = "encoder",
  155|         use_entmax: bool = False,
  156|         attention_grad_checkpoint: bool = False,
  157|         lag_bias_init: str = "normal",
  158|         alibi_slope_scale: float = 1.0,
  159|         query_uses_logvar: bool = False,
  160|         prior_availability_input: bool = False,
  161|         causal_norm: bool = False,
  162|         coverage_floor: float = 0.9,
  163|         base_decode: str = "sample",
  164|         persistence_residual: bool = False,
  165|         horizon_weight_halflife_steps: Optional[float] = None,
  166|         target_keep_index: Optional[Sequence[int]] = None,
  167|         target_delays: Optional[Sequence[int]] = None,
  168|         source_keep_index: Optional[Sequence[int]] = None,
  169|         source_delays: Optional[Sequence[int]] = None,
  170|         init_weights: bool = True,
  171|     ) -> None:
  360|         super().__init__()
  361|
  367|         if int(c_y) < 1 or int(c_u) < 1:
  368|             raise ValueError(
  369|                 f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
  370|             )
  371|         if int(num_heads) * int(d_head) != int(d_model):
  372|             raise ValueError(
  373|                 f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
  374|             )
  378|         if int(max_lag) < 0:
  379|             raise ValueError(f"max_lag must be >= 0, got {max_lag}")
  382|         if int(d_z) % int(num_heads) != 0:
  383|             raise ValueError(
  384|                 f"the head-structured latent requires d_z % num_heads == 0, "
  385|                 f"got d_z={d_z}, num_heads={num_heads}"
  386|             )
  387|
  391|         self.geometry = TrimmedRawGeometry(
  392|             raw_len=int(sequence_length) * int(raw_per_step),
  393|             decimation=int(raw_per_step),
  394|             horizon=int(horizon),
  395|             warmup=int(warmup_period),
  396|         )
  397|
  398|         self.sequence_length = int(sequence_length)
  399|         self.d_model = int(d_model)
  400|         self.d_z = int(d_z)
  401|         self.horizon = int(horizon)
  402|         self.raw_per_step = int(raw_per_step)
  403|         self.warmup_period = int(warmup_period)
  404|         self.c_y = int(c_y)
  405|         self.c_u = int(c_u)
  406|         self.use_up_st = bool(use_up_st)
  407|         self.max_lag = int(max_lag)
  408|         self.num_heads = int(num_heads)
  409|         self.mu_scale = float(mu_scale)
  410|         self.delta_mu_scale = float(delta_mu_scale)
  411|         self.delta_logvar_scale = float(delta_logvar_scale)
  412|         self.posterior_logvar_mode = validate_choice(
  413|             posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
  414|         )
  427|         self.source_dropout = float(dropout if source_dropout is None else source_dropout)
  428|         self.posterior_source_dropout = 0.0 if source_dropout is None else float(source_dropout)
  429|         self.posterior_event_gate = bool(posterior_event_gate)
  430|         self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
  431|         self.coverage_floor = float(coverage_floor)
  432|         self.base_decode = validate_choice(base_decode, BASE_DECODE_CHOICES, "base_decode")
  437|         self.persistence_residual = bool(persistence_residual)
  438|         if self.persistence_residual:
  439|             self._check_persistence_target()
  443|         self.horizon_embed_std = float(horizon_embed_std)
  444|         self.head_init_calibration = bool(head_init_calibration)
  445|         self.a_head_gain = float(a_head_gain)
  446|
  451|         self.register_buffer(
  452|             "future_index", build_future_index(self.geometry), persistent=False
  453|         )
  454|
  467|         self.horizon_weight_halflife_steps = (
  468|             None
  469|             if horizon_weight_halflife_steps is None
  470|             else float(horizon_weight_halflife_steps)
  471|         )
  472|         if self.horizon_weight_halflife_steps is not None:
  473|             self.register_buffer(
  474|                 "horizon_weight",
  475|                 horizon_decay_weight(self.horizon_weight_halflife_steps, self.horizon),
  476|                 persistent=False,
  477|             )
  478|
  483|         self.target_gate = self._build_channel_gate(
  484|             self.c_y, target_keep_index, target_delays
  485|         )
  486|         self.source_gate = self._build_channel_gate(
  487|             self.c_u, source_keep_index, source_delays
  488|         )
  508|         self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)
  509|         self.source_adapter = self._build_adapter(
  510|             self.source_gate, self.c_u, self.source_dropout
  511|         )
  512|
  522|         extra_dilations = tuple(int(x) for x in encoder_extra_dilations)
  523|         extra_kernels = tuple(int(encoder_extra_kernel) for _ in extra_dilations)
  524|         encoder_dilations = (1, 2, 4) + extra_dilations
  525|         self.target_encoder = CausalConvLstmEncoder(
  526|             d_model=d_model,
  527|             cnn_kernels=(3, 7, 11) + extra_kernels,
  528|             cnn_dilations=encoder_dilations,
  529|             lstm_layers=lstm_layers,
  530|             lstm_dropout=dropout,
  531|             conv_dropout=dropout,
  532|             stack_skip_connection=False,
  533|             post_residual_activation=False,
  534|             conv_norm_groups=conv_norm_groups,
  535|         )
  540|         self.lag_kv_source = validate_choice(
  541|             lag_kv_source, LAG_KV_SOURCE_CHOICES, "lag_kv_source"
  542|         )
  543|         if self.lag_kv_source == "encoder":
  544|             self.source_encoder = CausalConvLstmEncoder(
  545|                 d_model=d_model,
  546|                 cnn_kernels=(3, 5, 11) + extra_kernels,
  547|                 cnn_dilations=encoder_dilations,
  548|                 lstm_layers=lstm_layers,
  549|                 lstm_dropout=self.source_dropout,
  550|                 conv_dropout=self.source_dropout,
  551|                 stack_skip_connection=False,
  552|                 post_residual_activation=False,
  553|                 conv_norm_groups=conv_norm_groups,
  554|             )
  555|         elif self.lag_kv_source == "conv_stem":
  565|             self.source_kv_stem = CausalConvStem(
  566|                 d_model=d_model,
  567|                 cnn_kernels=(3, 5, 11) + extra_kernels,
  568|                 cnn_dilations=encoder_dilations,
  569|                 conv_dropout=self.source_dropout,
  570|                 conv_norm_groups=conv_norm_groups,
  571|             )
  572|
  576|         self.prior_availability_input = bool(prior_availability_input)
  577|         self.prior_head = FullLatentPriorHead(
  578|             d_model=d_model,
  579|             d_z=d_z,
  580|             logvar_clamp=logvar_clamp,
  581|             dropout=dropout,
  582|             mu_scale=self.mu_scale,
  583|             clock_dim=self._prior_clock_dim() if self.prior_availability_input else None,
  584|         )
  585|
  592|         self.query_uses_logvar = bool(query_uses_logvar)
  593|         query_in = 2 * self.d_z if self.query_uses_logvar else self.d_z
  594|         self.query_proj = nn.Linear(query_in, d_model)
  595|
  599|         self.lag_attn = LagCrossAttention(
  600|             d_model=d_model,
  601|             num_heads=num_heads,
  602|             d_head=d_head,
  603|             max_lag=max_lag,
  604|             dropout=0.0,
  605|             use_entmax=use_entmax,
  606|             grad_checkpoint=attention_grad_checkpoint,
  607|             lag_bias_init=lag_bias_init,
  608|             alibi_slope_scale=alibi_slope_scale,
  609|         )
  610|         self.posterior_head = PosteriorHead(
  611|             d_model=d_model,
  612|             d_z=d_z,
  613|             logvar_clamp=logvar_clamp,
  614|             dropout=dropout,
  615|             delta_mu_scale=self.delta_mu_scale,
  616|             head_structured=True,
  617|             num_heads=num_heads,
  618|             d_head=d_head,
  619|             delta_logvar_scale=self.delta_logvar_scale,
  620|             posterior_logvar_mode=self.posterior_logvar_mode,
  623|             source_dropout=self.posterior_source_dropout,
  624|             event_gate=self.posterior_event_gate,
  625|         )
  626|         self.te_analysis = TEAnalysisHead()
  627|
  645|         self.horizon_core = HorizonDecoderCore(
  646|             d_hidden=decoder_hidden,
  647|             horizon=horizon,
  648|             kernel_size=horizon_kernel,
  649|             depth=horizon_depth,
  650|             film=horizon_film,
  651|             film_per_block=True,
  652|             attention_blocks=horizon_attention_blocks,
  653|         )
  654|         self.decoder_out_channels = (
  655|             self._default_decoder_out_channels()
  656|             if decoder_out_channels is None
  657|             else int(decoder_out_channels)
  658|         )
  659|         self.decoder = BaselineFutureDecoder(
  660|             core=self.horizon_core,
  661|             d_model=d_z,
  662|             out_channels=self.decoder_out_channels,
  663|             d_hidden=decoder_hidden,
  664|             dropout=0.0,
  665|             logvar_clamp=logvar_clamp,
  668|             persistence_residual=self.persistence_residual,
  669|         )
  670|
  671|         self.causal_norm = bool(causal_norm)
  672|         if self.causal_norm:
  677|             source_body = self.source_kv_body()
  678|             self.n_causalized_norms = causalize_norms(self.target_encoder) + (
  679|                 0 if source_body is None else causalize_norms(source_body)
  680|             )
  681|         else:
  682|             self.n_causalized_norms = 0
  683|
  687|         for parameter in self.lag_attn.W_o.parameters():
  688|             parameter.requires_grad_(False)
  689|
  690|         if init_weights:
  691|             initialization(self)
  694|         self._zero_init_delta_heads()
  699|         self.prior_head.zero_init_clock()
  703|         self.posterior_head.zero_init_event_gate()
  710|         self._zero_init_film_generators()
  711|
  716|         if self.horizon_embed_std != 0.02:
  717|             self._reinit_horizon_embedding()
  718|         if self.head_init_calibration:
  719|             self._calibrate_output_heads()
  720|             self._calibrate_prior_scale()
  721|         if self.a_head_gain != 1.0:
  722|             self._set_a_head_gain()
  723|
  724|     def _default_decoder_out_channels(self) -> int:
  743|         return self.raw_per_step
  744|
  745|     @staticmethod
  746|     def _build_channel_gate(
  747|         declared_width: int,
  748|         keep_index: Optional[Sequence[int]],
  749|         delays: Optional[Sequence[int]],
  750|     ) -> Optional[ChannelGate]:
  770|         if keep_index is None and delays is None:
  771|             return None
  772|         return ChannelGate(
  773|             declared_width=int(declared_width), keep_index=keep_index, delays=delays
  774|         )
  775|
  776|     def _build_adapter(
  777|         self, gate: Optional[ChannelGate], declared_width: int, dropout: float
  778|     ) -> AvailabilityInputAdapter:
  795|         width = declared_width if gate is None else gate.out_channels
  796|         delays = None if gate is None else [int(value) for value in gate.delay.delay_steps]
  797|         return AvailabilityInputAdapter(
  798|             in_dim=width,
  799|             d_model=self.d_model,
  800|             sequence_length=self.sequence_length,
  801|             dropout=dropout,
  802|             delays=delays,
  803|         )
  804|
  805|     def source_kv_body(self) -> Optional[nn.Module]:
  817|         if self.lag_kv_source == "encoder":
  818|             return self.source_encoder
  819|         if self.lag_kv_source == "conv_stem":
  820|             return self.source_kv_stem
  821|         return None
  822|
  823|     def source_kv_modules(self) -> Tuple[nn.Module, ...]:
  834|         body = self.source_kv_body()
  835|         return (self.source_adapter,) if body is None else (self.source_adapter, body)
  836|
  837|     def encode_source_kv(self, source: torch.Tensor) -> torch.Tensor:
  848|         encoded = source
  849|         for module in self.source_kv_modules():
  850|             encoded = module(encoded)
  851|         return encoded
  852|
  853|     def build_lag_mask(
  854|         self, seq_len: int, device: Optional[torch.device] = None
  855|     ) -> torch.Tensor:
  872|         return self.lag_attn.build_lag_mask(seq_len, device=device)
  873|
  874|     @property
  875|     def source_delay_steps(self) -> int:
  900|         return 0 if self.source_gate is None else self.source_gate.max_delay
  901|
  902|     @staticmethod
  903|     def _zero_linear(layer: nn.Linear) -> None:
  905|         nn.init.zeros_(layer.weight)
  906|         if layer.bias is not None:
  907|             nn.init.zeros_(layer.bias)
  908|
  909|     def _prior_clock_dim(self) -> int:
  931|         raise ValueError(
  932|             f"prior_availability_input=True on {type(self).__name__}, whose input streams carry no "
  933|             f"warm-up: every channel is honest at every step, so nothing arrives late and there is "
  934|             f"no availability term in the KL to cancel. The flag belongs to the causal cells, whose "
  935|             f"one-sided inputs arrive over the first steps of a segment."
  936|         )
  937|
  938|     def _check_persistence_target(self) -> None:
  960|         raise ValueError(
  961|             f"persistence_residual=True on {type(self).__name__}, whose forecast block is R raw "
  962|             f"samples of one signal per horizon token: there is no per-channel level for a "
  963|             f"persistence term to carry forward. The flag belongs to the feature-target cells, "
  964|             f"whose block's last axis counts stored coefficients of the target itself."
  965|         )
  966|
  967|     def _zero_init_delta_heads(self) -> None:
  978|         for module in (self.posterior_head.delta_mu_head, self.posterior_head.delta_logvar_head):
  981|             if module is None:
  982|                 continue
  983|             layers = list(module) if isinstance(module, nn.ModuleList) else [module]
  984|             for layer in layers:
  985|                 self._zero_linear(cast(nn.Linear, layer))
  986|
  994|         independent = self.posterior_head.logvar_post_head
  995|         if independent is not None:
  998|             lo, hi = self.posterior_head.logvar_clamp
  999|             if not lo < 0.0 < hi:
 1000|                 raise ValueError(
 1001|                     f"posterior_logvar_mode='independent' seeds the head at unit scale, which "
 1002|                     f"needs 0 inside logvar_clamp; got ({lo}, {hi})"
 1003|                 )
 1004|             bias_value = math.log((0.0 - lo) / (hi - 0.0))
 1005|             layers = (
 1006|                 list(independent)
 1007|                 if isinstance(independent, nn.ModuleList)
 1008|                 else [independent]
 1009|             )
 1010|             for layer in layers:
 1011|                 linear = cast(nn.Linear, layer)
 1012|                 nn.init.zeros_(linear.weight)
 1013|                 linear.bias.data.fill_(bias_value)
 1014|
 1015|     def _zero_init_film_generators(self) -> None:
 1028|         core = self.horizon_core
 1029|         film_layers: list[nn.Module] = []
 1030|         if core.film_gen is not None:
 1031|             film_layers.append(core.film_gen)
 1032|         if core.refine.film is not None:
 1033|             film_layers.extend(core.refine.film)
 1034|         for layer in film_layers:
 1035|             self._zero_linear(cast(nn.Linear, layer))
 1036|
 1037|     def _reinit_horizon_embedding(self) -> None:
 1048|         nn.init.normal_(
 1049|             self.horizon_core.horizon_embedding, mean=0.0, std=self.horizon_embed_std
 1050|         )
 1051|
 1052|     def _calibrate_output_heads(self) -> None:
 1068|         self.decoder.mean_head.weight.data.mul_(0.02)
 1069|         self.decoder.logvar_head.bias.data.fill_(math.log(5.0 / 3.0))
 1070|         self.decoder.logvar_head.weight.data.mul_(0.1)
 1071|
 1072|     def _calibrate_prior_scale(self) -> None:
 1098|         lo, hi = self.logvar_clamp
 1099|         if not lo < 0.0 < hi:
 1100|             raise ValueError(
 1101|                 f"prior scale calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
 1102|             )
 1103|         head = self.prior_head.logvar_prior_head
 1104|         if not isinstance(head.skip_proj, nn.Linear):
 1105|             raise ValueError(
 1106|                 "prior scale calibration requires a projected skip on the log-variance head; "
 1107|                 "with d_model == d_z the skip is an identity and the output cannot be pinned"
 1108|             )
 1109|         self._zero_linear(head.skip_proj)
 1110|         final = cast(nn.Linear, head.body[-1])
 1111|         nn.init.zeros_(final.weight)
 1112|         final.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))
 1113|
 1114|     def _set_a_head_gain(self) -> None:
 1126|         nn.init.constant_(self.posterior_head.a_head_norm.weight, self.a_head_gain)
 1127|
 1128|     def _reparameterize_shared(
 1129|         self,
 1130|         mu_prior: torch.Tensor,
 1131|         logvar_prior: torch.Tensor,
 1132|         mu_post: torch.Tensor,
 1133|         logvar_post: torch.Tensor,
 1134|     ) -> Tuple[torch.Tensor, torch.Tensor]:
 1161|         epsilon = torch.randn_like(mu_prior)
 1162|         z_post = mu_post + epsilon * torch.exp(0.5 * logvar_post)
 1163|         if self.base_decode == "mean":
 1167|             return mu_prior, z_post
 1168|         z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
 1169|         return z_prior, z_post
 1170|
 1171|     def forward(
 1172|         self,
 1173|         y_st: torch.Tensor,
 1174|         y_ph: torch.Tensor,
 1175|         u_stream: torch.Tensor,
 1176|     ) -> Dict[str, torch.Tensor]:
 1225|         target = torch.cat([y_st, y_ph], dim=-1)
 1226|         if self.target_gate is not None:
 1227|             target = self.target_gate(target)
 1228|         source = u_stream if self.source_gate is None else self.source_gate(u_stream)
 1229|
 1230|         h_y = self.target_encoder(self.target_adapter(target))
 1231|         h_u = self.encode_source_kv(source)
 1232|
 1233|         mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(h_y)
 1234|
 1238|         query = (
 1239|             torch.cat([mu_prior, logvar_prior], dim=-1)
 1240|             if self.query_uses_logvar
 1241|             else mu_prior
 1242|         )
 1243|         _, alpha, attended_heads = self.lag_attn(
 1244|             self.query_proj(query), h_u, self.build_lag_mask(h_u.shape[1], h_u.device)
 1245|         )
 1246|
 1250|         event_gate = self.posterior_head.event_gate_values(attended_heads)
 1251|         mu_post, logvar_post = self.posterior_head(
 1252|             h_y, attended_heads, mu_prior, raw_logvar_prior, gate=event_gate
 1253|         )
 1254|         z_prior, z_post = self._reparameterize_shared(
 1255|             mu_prior, logvar_prior, mu_post, logvar_post
 1256|         )
 1257|
 1260|         with torch.no_grad():
 1261|             mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
 1262|             delta_mu_sat_frac = (
 1263|                 (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
 1264|             ).float().mean()
 1265|
 1269|         t_valid = self.geometry.t_valid
 1270|         mu_base, logvar_base = self.decoder(z_prior[:, :t_valid])
 1271|         mu_full, logvar_full = self.decoder(z_post[:, :t_valid])
 1272|
 1276|         kld_btd = self.kld_tensor(
 1277|             mu_prior=mu_prior,
 1278|             logvar_prior=logvar_prior,
 1279|             mu_post=mu_post,
 1280|             logvar_post=logvar_post,
 1281|         )
 1282|         kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
 1283|             kld_btd, alpha, head_structured=True
 1284|         )
 1285|
 1286|         outputs = {
 1287|             "mu_prior": mu_prior,
 1288|             "logvar_prior": logvar_prior,
 1289|             "raw_logvar_prior": raw_logvar_prior,
 1290|             "mu_post": mu_post,
 1291|             "logvar_post": logvar_post,
 1292|             "z_prior": z_prior,
 1293|             "z_post": z_post,
 1294|             "target_state": h_y,
 1295|             "source_state": h_u,
 1296|             "attended_source_heads": attended_heads,
 1297|             "attn_weights": alpha,
 1298|             "mu_base": mu_base,
 1299|             "logvar_base": logvar_base,
 1300|             "mu_full": mu_full,
 1301|             "logvar_full": logvar_full,
 1302|             "kld_per_t": kld_per_t,
 1303|             "kld_per_t_per_head": kld_per_t_per_head,
 1304|             "source_kl_lag_map": source_kl_lag_map,
 1305|             "mu_prior_sat_frac": mu_prior_sat_frac,
 1306|             "delta_mu_sat_frac": delta_mu_sat_frac,
 1307|         }
 1308|         if event_gate is not None:
 1311|             outputs["event_gate"] = event_gate
 1312|         return outputs
 1313|
 1314|     def kld_tensor(
 1315|         self,
 1316|         mu_prior: torch.Tensor,
 1317|         logvar_prior: torch.Tensor,
 1318|         mu_post: torch.Tensor,
 1319|         logvar_post: torch.Tensor,
 1320|     ) -> torch.Tensor:
 1338|         return closed_form_kld(mu_prior, logvar_prior, mu_post, logvar_post)
 1339|
 1340|     def compute_loss(
 1341|         self,
 1342|         forward_outputs: Dict[str, torch.Tensor],
 1343|         fhr_raw: torch.Tensor,
 1344|         *,
 1345|         weight: torch.Tensor,
 1346|         beta: float = 1.0,
 1347|         beta_prior: float = 0.0,
 1348|         lambda_full: float = 1.0,
 1349|         lambda_base: float = 1.0,
 1350|         likelihood: str = "gaussian_nll",
 1351|         free_bits: float = 0.0,
 1352|         lambda_ms: float = 0.0,
 1353|         lambda_deriv: float = 0.0,
 1354|         lambda_boundary: float = 0.0,
 1355|         lambda_event_gate: float = 0.0,
 1356|         full_anchor_weight: Optional[torch.Tensor] = None,
 1357|     ) -> Dict[str, Any]:
 1429|         return compute_raw_objective(
 1430|             forward_outputs,
 1431|             build_future_target(fhr_raw, self.geometry, future_index=self.future_index),
 1432|             weight=weight,
 1433|             geometry=self.geometry,
 1435|             block_width=self.geometry.r,
 1436|             coverage_floor=self.coverage_floor,
 1437|             logvar_clamp=self.logvar_clamp,
 1438|             beta=beta,
 1439|             beta_prior=beta_prior,
 1440|             lambda_full=lambda_full,
 1441|             lambda_base=lambda_base,
 1442|             likelihood=likelihood,
 1443|             free_bits=free_bits,
 1444|             lambda_ms=lambda_ms,
 1445|             lambda_deriv=lambda_deriv,
 1446|             lambda_boundary=lambda_boundary,
 1447|             lambda_event_gate=lambda_event_gate,
 1448|             full_anchor_weight=full_anchor_weight,
 1453|             horizon_weight=getattr(self, "horizon_weight", None),
 1454|         )
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
  158|         posterior_event_gate: bool = False,
  159|         lag_kv_source: str = "encoder",
  160|         use_entmax: bool = False,
  161|         attention_grad_checkpoint: bool = False,
  162|         lag_bias_init: str = "normal",
  163|         alibi_slope_scale: float = 1.0,
  164|         query_uses_logvar: bool = False,
  165|         prior_availability_input: bool = False,
  166|         coverage_floor: float = 0.9,
  167|         base_decode: str = "sample",
  168|         persistence_residual: bool = False,
  169|         horizon_weight_halflife_steps: Optional[float] = None,
  170|         target_keep_index: Optional[Sequence[int]] = None,
  171|         target_delays: Optional[Sequence[int]] = None,
  172|         source_keep_index: Optional[Sequence[int]] = None,
  173|         source_delays: Optional[Sequence[int]] = None,
  174|         init_weights: bool = True,
  175|     ) -> None:
  353|         super().__init__()
  354|
  360|         if int(c_y) < 1 or int(c_u) < 1:
  361|             raise ValueError(
  362|                 f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
  363|             )
  364|         if int(num_heads) * int(d_head) != int(d_model):
  365|             raise ValueError(
  366|                 f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
  367|             )
  371|         if int(max_lag) < 0:
  372|             raise ValueError(f"max_lag must be >= 0, got {max_lag}")
  375|         if int(d_z) % int(num_heads) != 0:
  376|             raise ValueError(
  377|                 f"the head-structured latent requires d_z % num_heads == 0, "
  378|                 f"got d_z={d_z}, num_heads={num_heads}"
  379|             )
  380|
  384|         self.geometry = TrimmedRawGeometry(
  385|             raw_len=int(sequence_length) * int(raw_per_step),
  386|             decimation=int(raw_per_step),
  387|             horizon=int(horizon),
  388|             warmup=int(warmup_period),
  389|         )
  390|
  391|         self.sequence_length = int(sequence_length)
  392|         self.d_model = int(d_model)
  393|         self.d_z = int(d_z)
  394|         self.horizon = int(horizon)
  395|         self.raw_per_step = int(raw_per_step)
  396|         self.warmup_period = int(warmup_period)
  397|         self.c_y = int(c_y)
  398|         self.c_u = int(c_u)
  399|         self.use_up_st = bool(use_up_st)
  400|         self.max_lag = int(max_lag)
  401|         self.num_heads = int(num_heads)
  402|         self.mu_scale = float(mu_scale)
  403|         self.delta_mu_scale = float(delta_mu_scale)
  404|         self.delta_logvar_scale = float(delta_logvar_scale)
  405|         self.posterior_logvar_mode = validate_choice(
  406|             posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
  407|         )
  420|         self.source_dropout = float(dropout if source_dropout is None else source_dropout)
  421|         self.posterior_source_dropout = 0.0 if source_dropout is None else float(source_dropout)
  422|         self.posterior_event_gate = bool(posterior_event_gate)
  423|         self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
  424|         self.coverage_floor = float(coverage_floor)
  425|         self.base_decode = validate_choice(base_decode, BASE_DECODE_CHOICES, "base_decode")
  430|         self.persistence_residual = bool(persistence_residual)
  431|         if self.persistence_residual:
  432|             self._check_persistence_target()
  436|         self.horizon_embed_std = float(horizon_embed_std)
  437|         self.head_init_calibration = bool(head_init_calibration)
  438|         self.a_head_gain = float(a_head_gain)
  439|
  444|         self.register_buffer(
  445|             "future_index", build_future_index(self.geometry), persistent=False
  446|         )
  447|
  460|         self.horizon_weight_halflife_steps = (
  461|             None
  462|             if horizon_weight_halflife_steps is None
  463|             else float(horizon_weight_halflife_steps)
  464|         )
  465|         if self.horizon_weight_halflife_steps is not None:
  466|             self.register_buffer(
  467|                 "horizon_weight",
  468|                 horizon_decay_weight(self.horizon_weight_halflife_steps, self.horizon),
  469|                 persistent=False,
  470|             )
  471|
  476|         self.target_gate = self._build_channel_gate(
  477|             self.c_y, target_keep_index, target_delays
  478|         )
  479|         self.source_gate = self._build_channel_gate(
  480|             self.c_u, source_keep_index, source_delays
  481|         )
  482|
  487|         self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)
  488|         self.source_adapter = self._build_adapter(
  489|             self.source_gate, self.c_u, self.source_dropout
  490|         )
  491|
  495|         self.target_encoder = CausalConvTransformerEncoder(
  496|             d_model=d_model,
  497|             sequence_length=self.sequence_length,
  498|             conv_kernels=encoder_conv_kernels,
  499|             conv_dilations=encoder_conv_dilations,
  500|             num_attention_blocks=target_attention_blocks,
  501|             num_heads=encoder_num_heads,
  502|             d_ff=encoder_d_ff,
  503|             attention_window=None,
  504|             dropout=dropout,
  505|         )
  511|         self.lag_kv_source = validate_choice(
  512|             lag_kv_source, LAG_KV_SOURCE_CHOICES, "lag_kv_source"
  513|         )
  514|         if self.lag_kv_source == "encoder":
  515|             self.source_encoder = CausalConvTransformerEncoder(
  516|                 d_model=d_model,
  517|                 sequence_length=self.sequence_length,
  518|                 conv_kernels=encoder_conv_kernels,
  519|                 conv_dilations=encoder_conv_dilations,
  520|                 num_attention_blocks=source_attention_blocks,
  521|                 num_heads=encoder_num_heads,
  522|                 d_ff=encoder_d_ff,
  523|                 attention_window=source_attention_window,
  524|                 dropout=self.source_dropout,
  525|             )
  526|         elif self.lag_kv_source == "conv_stem":
  529|             self.source_kv_stem = GatedCausalConvStem(
  530|                 d_model=d_model,
  531|                 conv_kernels=encoder_conv_kernels,
  532|                 conv_dilations=encoder_conv_dilations,
  533|                 dropout=self.source_dropout,
  534|             )
  535|
  539|         self.prior_availability_input = bool(prior_availability_input)
  540|         self.prior_head = FullLatentPriorHead(
  541|             d_model=d_model,
  542|             d_z=d_z,
  543|             logvar_clamp=logvar_clamp,
  544|             dropout=dropout,
  545|             mu_scale=self.mu_scale,
  546|             clock_dim=self._prior_clock_dim() if self.prior_availability_input else None,
  547|         )
  548|
  553|         self.query_uses_logvar = bool(query_uses_logvar)
  554|         query_in = 2 * self.d_z if self.query_uses_logvar else self.d_z
  555|         self.query_proj = nn.Linear(query_in, d_model)
  556|
  560|         self.lag_attn = LagCrossAttention(
  561|             d_model=d_model,
  562|             num_heads=num_heads,
  563|             d_head=d_head,
  564|             max_lag=max_lag,
  565|             dropout=0.0,
  566|             use_entmax=use_entmax,
  567|             grad_checkpoint=attention_grad_checkpoint,
  568|             lag_bias_init=lag_bias_init,
  569|             alibi_slope_scale=alibi_slope_scale,
  570|         )
  571|         self.posterior_head = PosteriorHead(
  572|             d_model=d_model,
  573|             d_z=d_z,
  574|             logvar_clamp=logvar_clamp,
  575|             dropout=dropout,
  576|             delta_mu_scale=self.delta_mu_scale,
  577|             head_structured=True,
  578|             num_heads=num_heads,
  579|             d_head=d_head,
  580|             delta_logvar_scale=self.delta_logvar_scale,
  581|             posterior_logvar_mode=self.posterior_logvar_mode,
  584|             source_dropout=self.posterior_source_dropout,
  585|             event_gate=self.posterior_event_gate,
  586|         )
  587|         self.te_analysis = TEAnalysisHead()
  588|
  598|         self.horizon_core = HorizonDecoderCore(
  599|             d_hidden=decoder_hidden,
  600|             horizon=horizon,
  601|             kernel_size=horizon_kernel,
  602|             depth=horizon_depth,
  603|             film=horizon_film,
  604|             film_per_block=True,
  605|             attention_blocks=horizon_attention_blocks,
  606|         )
  607|         self.decoder_out_channels = self._default_decoder_out_channels()
  608|         self.decoder = BaselineFutureDecoder(
  609|             core=self.horizon_core,
  610|             d_model=d_z,
  611|             out_channels=self.decoder_out_channels,
  612|             d_hidden=decoder_hidden,
  613|             dropout=0.0,
  614|             logvar_clamp=logvar_clamp,
  617|             persistence_residual=self.persistence_residual,
  618|         )
  619|
  623|         for parameter in self.lag_attn.W_o.parameters():
  624|             parameter.requires_grad_(False)
  625|
  631|         self.n_depthwise_init = 0
  632|         if init_weights:
  633|             initialization(self)
  641|             self.n_depthwise_init = init_depthwise_(self)
  644|         self._zero_init_delta_heads()
  649|         self.prior_head.zero_init_clock()
  653|         self.posterior_head.zero_init_event_gate()
  659|         self._zero_init_film_generators()
  660|
  665|         if self.horizon_embed_std != 0.02:
  666|             self._reinit_horizon_embedding()
  667|         if self.head_init_calibration:
  668|             self._calibrate_output_heads()
  669|             self._calibrate_prior_scale()
  670|         if self.a_head_gain != 1.0:
  671|             self._set_a_head_gain()
  672|
  673|     def _default_decoder_out_channels(self) -> int:
  697|         return self.raw_per_step
  698|
  699|     @staticmethod
  700|     def _build_channel_gate(
  701|         declared_width: int,
  702|         keep_index: Optional[Sequence[int]],
  703|         delays: Optional[Sequence[int]],
  704|     ) -> Optional[ChannelGate]:
  724|         if keep_index is None and delays is None:
  725|             return None
  726|         return ChannelGate(
  727|             declared_width=int(declared_width), keep_index=keep_index, delays=delays
  728|         )
  729|
  730|     def _build_adapter(
  731|         self, gate: Optional[ChannelGate], declared_width: int, dropout: float
  732|     ) -> AvailabilityInputAdapter:
  749|         width = declared_width if gate is None else gate.out_channels
  750|         delays = None if gate is None else [int(value) for value in gate.delay.delay_steps]
  751|         return AvailabilityInputAdapter(
  752|             in_dim=width,
  753|             d_model=self.d_model,
  754|             sequence_length=self.sequence_length,
  755|             dropout=dropout,
  756|             delays=delays,
  757|         )
  758|
  759|     def source_kv_body(self) -> Optional[nn.Module]:
  771|         if self.lag_kv_source == "encoder":
  772|             return self.source_encoder
  773|         if self.lag_kv_source == "conv_stem":
  774|             return self.source_kv_stem
  775|         return None
  776|
  777|     def source_kv_modules(self) -> Tuple[nn.Module, ...]:
  788|         body = self.source_kv_body()
  789|         return (self.source_adapter,) if body is None else (self.source_adapter, body)
  790|
  791|     def encode_source_kv(self, source: torch.Tensor) -> torch.Tensor:
  802|         encoded = source
  803|         for module in self.source_kv_modules():
  804|             encoded = module(encoded)
  805|         return encoded
  806|
  807|     def build_lag_mask(
  808|         self, seq_len: int, device: Optional[torch.device] = None
  809|     ) -> torch.Tensor:
  826|         return self.lag_attn.build_lag_mask(seq_len, device=device)
  827|
  828|     @property
  829|     def source_delay_steps(self) -> int:
  854|         return 0 if self.source_gate is None else self.source_gate.max_delay
  855|
  856|     @staticmethod
  857|     def _zero_linear(layer: nn.Linear) -> None:
  859|         nn.init.zeros_(layer.weight)
  860|         if layer.bias is not None:
  861|             nn.init.zeros_(layer.bias)
  862|
  863|     def _prior_clock_dim(self) -> int:
  885|         raise ValueError(
  886|             f"prior_availability_input=True on {type(self).__name__}, whose input streams carry no "
  887|             f"warm-up: every channel is honest at every step, so nothing arrives late and there is "
  888|             f"no availability term in the KL to cancel. The flag belongs to the causal cells, whose "
  889|             f"one-sided inputs arrive over the first steps of a segment."
  890|         )
  891|
  892|     def _check_persistence_target(self) -> None:
  914|         raise ValueError(
  915|             f"persistence_residual=True on {type(self).__name__}, whose forecast block is R raw "
  916|             f"samples of one signal per horizon token: there is no per-channel level for a "
  917|             f"persistence term to carry forward. The flag belongs to the feature-target cells, "
  918|             f"whose block's last axis counts stored coefficients of the target itself."
  919|         )
  920|
  921|     def _zero_init_delta_heads(self) -> None:
  932|         for module in (self.posterior_head.delta_mu_head, self.posterior_head.delta_logvar_head):
  935|             if module is None:
  936|                 continue
  937|             layers = list(module) if isinstance(module, nn.ModuleList) else [module]
  938|             for layer in layers:
  939|                 self._zero_linear(cast(nn.Linear, layer))
  940|
  948|         independent = self.posterior_head.logvar_post_head
  949|         if independent is not None:
  952|             lo, hi = self.posterior_head.logvar_clamp
  953|             if not lo < 0.0 < hi:
  954|                 raise ValueError(
  955|                     f"posterior_logvar_mode='independent' seeds the head at unit scale, which "
  956|                     f"needs 0 inside logvar_clamp; got ({lo}, {hi})"
  957|                 )
  958|             bias_value = math.log((0.0 - lo) / (hi - 0.0))
  959|             layers = (
  960|                 list(independent)
  961|                 if isinstance(independent, nn.ModuleList)
  962|                 else [independent]
  963|             )
  964|             for layer in layers:
  965|                 linear = cast(nn.Linear, layer)
  966|                 nn.init.zeros_(linear.weight)
  967|                 linear.bias.data.fill_(bias_value)
  968|
  969|     def _zero_init_film_generators(self) -> None:
  980|         core = self.horizon_core
  981|         film_layers: List[nn.Module] = []
  982|         if core.film_gen is not None:
  983|             film_layers.append(core.film_gen)
  984|         if core.refine.film is not None:
  985|             film_layers.extend(core.refine.film)
  986|         for layer in film_layers:
  987|             self._zero_linear(cast(nn.Linear, layer))
  988|
  989|     def _reinit_horizon_embedding(self) -> None:
 1000|         nn.init.normal_(
 1001|             self.horizon_core.horizon_embedding, mean=0.0, std=self.horizon_embed_std
 1002|         )
 1003|
 1004|     def _calibrate_output_heads(self) -> None:
 1020|         self.decoder.mean_head.weight.data.mul_(0.02)
 1021|         self.decoder.logvar_head.bias.data.fill_(math.log(5.0 / 3.0))
 1022|         self.decoder.logvar_head.weight.data.mul_(0.1)
 1023|
 1024|     def _calibrate_prior_scale(self) -> None:
 1050|         lo, hi = self.logvar_clamp
 1051|         if not lo < 0.0 < hi:
 1052|             raise ValueError(
 1053|                 f"prior scale calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
 1054|             )
 1055|         head = self.prior_head.logvar_prior_head
 1056|         if not isinstance(head.skip_proj, nn.Linear):
 1057|             raise ValueError(
 1058|                 "prior scale calibration requires a projected skip on the log-variance head; "
 1059|                 "with d_model == d_z the skip is an identity and the output cannot be pinned"
 1060|             )
 1061|         self._zero_linear(head.skip_proj)
 1062|         final = cast(nn.Linear, head.body[-1])
 1063|         nn.init.zeros_(final.weight)
 1064|         final.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))
 1065|
 1066|     def _set_a_head_gain(self) -> None:
 1078|         nn.init.constant_(self.posterior_head.a_head_norm.weight, self.a_head_gain)
 1079|
 1080|     def _reparameterize_shared(
 1081|         self,
 1082|         mu_prior: torch.Tensor,
 1083|         logvar_prior: torch.Tensor,
 1084|         mu_post: torch.Tensor,
 1085|         logvar_post: torch.Tensor,
 1086|     ) -> Tuple[torch.Tensor, torch.Tensor]:
 1114|         epsilon = torch.randn_like(mu_prior)
 1115|         z_post = mu_post + epsilon * torch.exp(0.5 * logvar_post)
 1116|         if self.base_decode == "mean":
 1120|             return mu_prior, z_post
 1121|         z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
 1122|         return z_prior, z_post
 1123|
 1124|     def forward(
 1125|         self,
 1126|         y_st: torch.Tensor,
 1127|         y_ph: torch.Tensor,
 1128|         u_stream: torch.Tensor,
 1129|     ) -> Dict[str, torch.Tensor]:
 1177|         target = torch.cat([y_st, y_ph], dim=-1)
 1178|         if self.target_gate is not None:
 1179|             target = self.target_gate(target)
 1180|         source = u_stream if self.source_gate is None else self.source_gate(u_stream)
 1181|
 1182|         h_y = self.target_encoder(self.target_adapter(target))
 1183|         h_u = self.encode_source_kv(source)
 1184|
 1185|         mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(h_y)
 1186|
 1190|         query = (
 1191|             torch.cat([mu_prior, logvar_prior], dim=-1)
 1192|             if self.query_uses_logvar
 1193|             else mu_prior
 1194|         )
 1195|         _, alpha, attended_heads = self.lag_attn(
 1196|             self.query_proj(query), h_u, self.build_lag_mask(h_u.shape[1], h_u.device)
 1197|         )
 1198|
 1202|         event_gate = self.posterior_head.event_gate_values(attended_heads)
 1203|         mu_post, logvar_post = self.posterior_head(
 1204|             h_y, attended_heads, mu_prior, raw_logvar_prior, gate=event_gate
 1205|         )
 1206|         z_prior, z_post = self._reparameterize_shared(
 1207|             mu_prior, logvar_prior, mu_post, logvar_post
 1208|         )
 1209|
 1212|         with torch.no_grad():
 1213|             mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
 1214|             delta_mu_sat_frac = (
 1215|                 (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
 1216|             ).float().mean()
 1217|
 1220|         t_valid = self.geometry.t_valid
 1221|         mu_base, logvar_base = self.decoder(z_prior[:, :t_valid])
 1222|         mu_full, logvar_full = self.decoder(z_post[:, :t_valid])
 1223|
 1227|         kld_btd = self.kld_tensor(
 1228|             mu_prior=mu_prior,
 1229|             logvar_prior=logvar_prior,
 1230|             mu_post=mu_post,
 1231|             logvar_post=logvar_post,
 1232|         )
 1233|         kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
 1234|             kld_btd, alpha, head_structured=True
 1235|         )
 1236|
 1237|         outputs = {
 1238|             "mu_prior": mu_prior,
 1239|             "logvar_prior": logvar_prior,
 1240|             "raw_logvar_prior": raw_logvar_prior,
 1241|             "mu_post": mu_post,
 1242|             "logvar_post": logvar_post,
 1243|             "z_prior": z_prior,
 1244|             "z_post": z_post,
 1245|             "target_state": h_y,
 1246|             "source_state": h_u,
 1247|             "attended_source_heads": attended_heads,
 1248|             "attn_weights": alpha,
 1249|             "mu_base": mu_base,
 1250|             "logvar_base": logvar_base,
 1251|             "mu_full": mu_full,
 1252|             "logvar_full": logvar_full,
 1253|             "kld_per_t": kld_per_t,
 1254|             "kld_per_t_per_head": kld_per_t_per_head,
 1255|             "source_kl_lag_map": source_kl_lag_map,
 1256|             "mu_prior_sat_frac": mu_prior_sat_frac,
 1257|             "delta_mu_sat_frac": delta_mu_sat_frac,
 1258|         }
 1259|         if event_gate is not None:
 1262|             outputs["event_gate"] = event_gate
 1263|         return outputs
 1264|
 1265|     def kld_tensor(
 1266|         self,
 1267|         mu_prior: torch.Tensor,
 1268|         logvar_prior: torch.Tensor,
 1269|         mu_post: torch.Tensor,
 1270|         logvar_post: torch.Tensor,
 1271|     ) -> torch.Tensor:
 1288|         return closed_form_kld(mu_prior, logvar_prior, mu_post, logvar_post)
 1289|
 1290|     def compute_loss(
 1291|         self,
 1292|         forward_outputs: Dict[str, torch.Tensor],
 1293|         fhr_raw: torch.Tensor,
 1294|         *,
 1295|         weight: torch.Tensor,
 1296|         beta: float = 1.0,
 1297|         beta_prior: float = 0.0,
 1298|         lambda_full: float = 1.0,
 1299|         lambda_base: float = 1.0,
 1300|         likelihood: str = "gaussian_nll",
 1301|         free_bits: float = 0.0,
 1302|         lambda_ms: float = 0.0,
 1303|         lambda_deriv: float = 0.0,
 1304|         lambda_boundary: float = 0.0,
 1305|         lambda_event_gate: float = 0.0,
 1306|         full_anchor_weight: Optional[torch.Tensor] = None,
 1307|     ) -> Dict[str, Any]:
 1350|         return compute_raw_objective(
 1351|             forward_outputs,
 1352|             build_future_target(fhr_raw, self.geometry, future_index=self.future_index),
 1353|             weight=weight,
 1354|             geometry=self.geometry,
 1356|             block_width=self.geometry.r,
 1357|             coverage_floor=self.coverage_floor,
 1358|             logvar_clamp=self.logvar_clamp,
 1359|             beta=beta,
 1360|             beta_prior=beta_prior,
 1361|             lambda_full=lambda_full,
 1362|             lambda_base=lambda_base,
 1363|             likelihood=likelihood,
 1364|             free_bits=free_bits,
 1365|             lambda_ms=lambda_ms,
 1366|             lambda_deriv=lambda_deriv,
 1367|             lambda_boundary=lambda_boundary,
 1368|             lambda_event_gate=lambda_event_gate,
 1369|             full_anchor_weight=full_anchor_weight,
 1374|             horizon_weight=getattr(self, "horizon_weight", None),
 1375|         )
```

## 11. Feature-target mixin — `teb_vae/lag_attn_fs/nets/feature_target.py` (`FeatureForecastTarget`)

Decoder width hook (kept target channels), the stored-clock target gather $Y^+[b,a,\tau,k]=Y[b,t_a+1+\tau,\mathrm{keep}[k]]$, `forecast_likelihood_kwargs` (the per-channel scored-horizon `cell_mask` and the AR(1) coefficient $\phi_c=\tanh(a_c)$, both `None` when not built), and `compute_loss` delegation with `block_width=C_keep` and the channel and horizon weights.

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
  105|     def forecast_likelihood_kwargs(self) -> Dict[str, Optional[torch.Tensor]]:
  119|         ar_logit = getattr(self, "target_ar_logit", None)
  120|         return {
  121|             "cell_mask": getattr(self, "target_cell_mask", None),
  122|             "ar_coef": None if ar_logit is None else torch.tanh(ar_logit),
  123|         }
  124|
  125|     def _default_decoder_out_channels(self) -> int:
  139|         return self.c_y if self.target_gate is None else self.target_gate.out_channels
  140|
  141|     def _build_forecast_target(
  142|         self, target_features: torch.Tensor, anchors: Optional[torch.Tensor] = None
  143|     ) -> torch.Tensor:
  179|         if target_features.dim() != 3:
  180|             raise ValueError(
  181|                 f"target stream must be 3-D (B, T, c_y), got shape {tuple(target_features.shape)}"
  182|             )
  183|         if target_features.size(1) != self.geometry.t:
  184|             raise ValueError(
  185|                 f"target stream length {target_features.size(1)} != geometry.t "
  186|                 f"{self.geometry.t}; this geometry assumes the trimmed loader "
  187|                 f"(trim_minutes: 1.0 -> T = {self.geometry.t} decimated steps), so a mismatch "
  188|                 f"means the loader ran at a different trim_minutes"
  189|             )
  190|         if target_features.size(2) != self.c_y:
  191|             raise ValueError(
  192|                 f"target stream has {target_features.size(2)} channels but the model declares "
  193|                 f"c_y={self.c_y}; the surviving-channel index is positional into the declared "
  194|                 f"width, so a mismatch would gather the wrong channels rather than fail"
  195|             )
  196|
  197|         gathered = (
  198|             target_features
  199|             if self.target_gate is None
  200|             else torch.index_select(target_features, -1, self.target_gate.keep_index)
  201|         )
  202|         if anchors is None:
  205|             return (
  206|                 gathered[:, 1:, :]
  207|                 .unfold(dimension=1, size=self.horizon, step=1)
  208|                 .permute(0, 1, 3, 2)
  209|             )
  210|
  211|         batch, channels = gathered.shape[0], gathered.shape[-1]
  212|         steps = torch.arange(self.horizon, device=gathered.device)
  214|         time_index = anchors.to(torch.long)[:, :, None] + 1 + steps[None, None, :]
  215|         window = gathered.gather(
  216|             1, time_index.reshape(batch, -1, 1).expand(-1, -1, channels)
  217|         )
  218|         return window.reshape(batch, anchors.shape[1], self.horizon, channels)
  219|
  348|     def compute_loss(
  349|         self,
  350|         forward_outputs: Dict[str, torch.Tensor],
  351|         target_features: torch.Tensor,
  352|         *,
  353|         weight: torch.Tensor,
  354|         beta: float = 1.0,
  355|         beta_prior: float = 0.0,
  356|         lambda_full: float = 1.0,
  357|         lambda_base: float = 1.0,
  358|         likelihood: str = "gaussian_nll",
  359|         free_bits: float = 0.0,
  360|         lambda_ms: float = 0.0,
  361|         lambda_deriv: float = 0.0,
  362|         lambda_boundary: float = 0.0,
  363|         lambda_event_gate: float = 0.0,
  364|         full_anchor_weight: Optional[torch.Tensor] = None,
  365|     ) -> Dict[str, Any]:
  428|         target = self._build_forecast_target(
  429|             target_features, forward_outputs.get("anchor_index")
  430|         )
  431|         result = compute_shared_objective(
  432|             forward_outputs,
  433|             target,
  434|             weight=weight,
  435|             geometry=self.geometry,
  438|             block_width=self.decoder_out_channels,
  439|             coverage_floor=self.coverage_floor,
  440|             logvar_clamp=self.logvar_clamp,
  441|             beta=beta,
  442|             beta_prior=beta_prior,
  443|             lambda_full=lambda_full,
  444|             lambda_base=lambda_base,
  445|             likelihood=likelihood,
  446|             free_bits=free_bits,
  447|             lambda_ms=lambda_ms,
  448|             lambda_deriv=lambda_deriv,
  449|             lambda_boundary=lambda_boundary,
  450|             lambda_event_gate=lambda_event_gate,
  451|             full_anchor_weight=full_anchor_weight,
  455|             channel_weight=getattr(self, "target_channel_weight", None),
  458|             horizon_weight=getattr(self, "horizon_weight", None),
  460|             **self.forecast_likelihood_kwargs(),
  461|         )
  465|         result["metrics"].update(
  466|             self._resolved_forecast_gaps(
  467|                 forward_outputs, target, weight, likelihood=likelihood
  468|             )
  469|         )
  470|         return result
```

## 12. Causal input mixin — `teb_vae/lag_attn_cfs/nets/causal_inputs.py` (`CausalWarmupInputs`)

Owns: `_set_causal_inputs`, `anchor_ceiling`, `_validate_causal_geometry`, `_combined_source_steps` ($W'_c+d_c$), `_prior_clock` (zeroed-source encode), `_prior_clock_dim`, `_build_adapter` override (availability at $W'_c+d_c$), `build_lag_mask` (floor), `_build_anchor_index` (tiling), the source-validity path (`_check_source_valid`, `source_key_mask`, `_source_kv_reach`, `_close_gate`), and the **tiled `forward` that all four models run**. `source_key_mask` gathers and delays a per-channel validity mask through the source gate, requires every kept channel to be valid at a step, widens a bad step by the K/V body's receptive field, and returns the key mask $K_{b,t,\ell}=\mathrm{lag\_mask}_{t,\ell}\wedge m_{b,t-\ell}$. `_close_gate` sets the effective gate $g_t\,o_t$ to zero where no source key is valid.

`teb_vae/lag_attn_cfs/nets/causal_inputs.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   73| from __future__ import annotations
   74|
   75| from typing import Dict, Optional, Sequence, Tuple, Union
   76|
   77| import torch
   78| import torch.nn.functional as F
   79|
   80| from teb_vae.lag_attn.nets.delays import ChannelGate
   81| from teb_vae.lag_attn.nets.encoders import AvailabilityInputAdapter
   82| from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
   83|
   96| CAUSAL_ONLY_KEYWORDS: Tuple[str, ...] = (
   97|     "target_warmup_steps",
   98|     "source_warmup_steps",
   99|     "anchor_stride",
  100|     "lag_floor",
  101|     "target_weight_st",
  102|     "target_weight_ph",
  103|     "target_align_delays",
  104|     "source_align_delays",
  105|     "target_novelty_frac",
  106|     "target_forecast_shift",
  107|     "target_scored_horizon",
  108|     "forecast_ar_residual",
  109|     "source_quality_mask",
  110| )
  111|
  115| FORWARDED_EXCLUSIONS: Tuple[str, ...] = ("self", "__class__") + CAUSAL_ONLY_KEYWORDS
  116|
  118| def _refuse_phase_outside(phase: torch.Tensor, stride: int) -> None:
  128|     outside = (phase < 0) | (phase >= stride)
  129|     if bool(outside.any()):
  130|         offending = int(phase[outside][0])
  131|         raise ValueError(
  132|             f"anchor_phase {offending} is outside [0, anchor_stride) = [0, {stride}). The "
  133|             f"anchor set truncates rather than rotating, so a phase at or above the stride "
  134|             f"drops leading anchors instead of shifting the grid -- and at stride 1 the only "
  135|             f"admissible phase is 0."
  136|         )
  137|
  139| class CausalWarmupInputs:
  165|
  166|     target_warmup_steps: Optional[Tuple[int, ...]]
  167|     source_warmup_steps: Optional[Tuple[int, ...]]
  168|     anchor_stride: int
  169|     lag_floor: int
  170|     target_forecast_shift: Optional[Tuple[int, ...]]
  171|
  175|     def _set_causal_inputs(
  176|         self,
  177|         *,
  178|         horizon: int,
  179|         target_keep_index: Optional[Sequence[int]],
  180|         target_warmup_steps: Optional[Sequence[int]],
  181|         source_keep_index: Optional[Sequence[int]],
  182|         source_warmup_steps: Optional[Sequence[int]],
  183|         anchor_stride: int,
  184|         lag_floor: int,
  185|         target_forecast_shift: Optional[Sequence[int]] = None,
  186|         source_quality_mask: bool = False,
  187|     ) -> None:
  224|         if not 1 <= int(anchor_stride) <= int(horizon):
  225|             raise ValueError(
  226|                 f"anchor_stride must be in [1, horizon] = [1, {int(horizon)}], got "
  227|                 f"{anchor_stride}. Below 1 there is no anchor set; above the horizon the decoded "
  228|                 f"windows leave gaps, so target steps between two tiles would never be scored at "
  229|                 f"any phase."
  230|             )
  231|         if int(lag_floor) < 0:
  232|             raise ValueError(f"lag_floor must be >= 0, got {lag_floor}")
  233|         self.source_quality_mask = bool(source_quality_mask)
  234|
  238|         for stream, warmup, keep in (
  239|             ("target", target_warmup_steps, target_keep_index),
  240|             ("source", source_warmup_steps, source_keep_index),
  241|         ):
  242|             if warmup is not None and keep is None:
  243|                 raise ValueError(
  244|                     f"{stream}_warmup_steps was given without {stream}_keep_index. The warm-up "
  245|                     f"vector is positional against the surviving channels, so the two are "
  246|                     f"resolved together and travel together; a keep-index covering every channel "
  247|                     f"is still an explicit keep-index."
  248|                 )
  249|
  256|         if target_forecast_shift is not None:
  257|             if target_keep_index is None:
  258|                 raise ValueError(
  259|                     "target_forecast_shift was given without target_keep_index. The shift is "
  260|                     "positional against the surviving target channels, so the two are resolved "
  261|                     "together and travel together."
  262|                 )
  263|             shifts = tuple(int(step) for step in target_forecast_shift)
  264|             if shifts and min(shifts) < 0 < max(shifts):
  265|                 raise ValueError(
  266|                     f"target_forecast_shift mixes signs (min {min(shifts)}, max {max(shifts)}). "
  267|                     f"Each forecast clock is single-signed -- 'physical' advances, 'input' delays "
  268|                     f"-- so a mixed vector names no clock and its two halves would be checked "
  269|                     f"against each other's geometry."
  270|                 )
  271|             target_forecast_shift = shifts
  272|
  273|         self.target_warmup_steps = None if target_warmup_steps is None else tuple(
  274|             int(step) for step in target_warmup_steps
  275|         )
  276|         self.source_warmup_steps = None if source_warmup_steps is None else tuple(
  277|             int(step) for step in source_warmup_steps
  278|         )
  279|         self.anchor_stride = int(anchor_stride)
  280|         self.lag_floor = int(lag_floor)
  281|         self.target_forecast_shift = target_forecast_shift
  282|
  283|     @property
  284|     def anchor_ceiling(self) -> int:
  297|         shift = getattr(self, "target_forecast_shift", None)
  298|         if not shift:
  299|             return int(self.geometry.t_valid)
  300|         return int(self.geometry.t_valid) - max(0, max(shift))
  301|
  302|     def _validate_causal_geometry(self) -> None:
  325|         if self.anchor_stride > self.anchor_ceiling - self.warmup_period:
  326|             advance = (
  327|                 ""
  328|                 if self.anchor_ceiling == self.geometry.t_valid
  329|                 else (
  330|                     f" (T_valid {self.geometry.t_valid} less the forecast clock's largest "
  331|                     f"advance {self.geometry.t_valid - self.anchor_ceiling})"
  332|                 )
  333|             )
  334|             raise ValueError(
  335|                 f"anchor_stride={self.anchor_stride} leaves no anchor at phase "
  336|                 f"{self.anchor_stride - 1}: the first would be "
  337|                 f"{self.warmup_period + self.anchor_stride - 1}, against an anchor ceiling of "
  338|                 f"{self.anchor_ceiling}{advance} and warmup_period={self.warmup_period}. A "
  339|                 f"sample drawn at that phase would contribute no forecast at all, and nothing "
  340|                 f"downstream reports an empty anchor row."
  341|             )
  342|         floor_args = [
  343|             self.warmup_period,
  344|             self.target_warmup_steps or (),
  345|             ()
  346|             if self.target_gate is None
  347|             else tuple(int(shift) for shift in self.target_gate.delay.delay_steps),
  348|         ]
  354|         if self.target_forecast_shift:
  355|             floor_args.append(self.target_forecast_shift)
  356|         self._check_anchor_floor(*floor_args)
  357|         self._resolve_warmup_readout_constants()
  358|
  359|     def _combined_source_steps(self) -> Optional[Tuple[int, ...]]:
  378|         waits = self.source_warmup_steps
  379|         if waits is None or self.source_gate is None:
  380|             return waits
  381|         return tuple(
  382|             wait + int(shift)
  383|             for wait, shift in zip(waits, self.source_gate.delay.delay_steps)
  384|         )
  385|
  386|     def _prior_clock(self, u_stream: torch.Tensor) -> torch.Tensor:
  444|         zeros = u_stream.new_zeros((1, *u_stream.shape[1:]))
  445|         gated = zeros if self.source_gate is None else self.source_gate(zeros)
  450|         modules = self.source_kv_modules()
  451|         was_training = [module.training for module in modules]
  452|         for module in modules:
  453|             module.eval()
  454|         try:
  455|             with torch.no_grad():
  456|                 encoded = self.encode_source_kv(gated)
  457|         finally:
  458|             for module, flag in zip(modules, was_training):
  459|                 module.train(flag)
  460|         return encoded.detach()
  461|
  462|     def _prior_clock_dim(self) -> int:
  474|         return int(self.d_model)
  475|
  626|     def _build_adapter(
  627|         self, gate: Optional[ChannelGate], declared_width: int, dropout: float
  628|     ) -> AvailabilityInputAdapter:
  664|         warmup = (
  665|             self.target_warmup_steps
  666|             if gate is self.target_gate
  667|             else self.source_warmup_steps
  668|         )
  669|         if warmup is None:
  670|             return super()._build_adapter(gate, declared_width, dropout)
  671|
  672|         width = declared_width if gate is None else gate.out_channels
  673|         delays = list(warmup)
  674|         if gate is not None:
  675|             delays = [
  676|                 wait + int(shift) for wait, shift in zip(delays, gate.delay.delay_steps)
  677|             ]
  678|         return AvailabilityInputAdapter(
  679|             in_dim=width,
  680|             d_model=self.d_model,
  681|             sequence_length=self.sequence_length,
  682|             dropout=dropout,
  683|             delays=delays,
  684|         )
  685|
  686|     def build_lag_mask(
  687|         self, seq_len: int, device: Optional[torch.device] = None
  688|     ) -> torch.Tensor:
  707|         mask = super().build_lag_mask(seq_len, device=device)
  708|         if self.lag_floor == 0:
  709|             return mask
  710|         steps = torch.arange(seq_len, device=device)[:, None]
  711|         lags = torch.arange(self.lag_attn.L, device=device)[None, :]
  712|         return mask & (steps - lags >= self.lag_floor)
  713|
  717|     def _build_anchor_index(
  718|         self,
  719|         batch: int,
  720|         device: torch.device,
  721|         anchor_phase: Optional[Union[int, torch.Tensor]] = None,
  722|         anchor_stride: Optional[int] = None,
  723|     ) -> Tuple[torch.Tensor, torch.Tensor]:
  764|         stride = self.anchor_stride if anchor_stride is None else int(anchor_stride)
  765|         if not 1 <= stride <= self.horizon:
  766|             raise ValueError(
  767|                 f"anchor_stride must be in [1, horizon] = [1, {self.horizon}], got {stride}"
  768|             )
  769|
  773|         floor, t_valid = self.warmup_period, self.anchor_ceiling
  774|         span = t_valid - floor
  775|
  776|         if anchor_phase is None:
  780|             if stride > 1:
  781|                 raise ValueError(
  782|                     f"anchor_phase is required at anchor_stride={stride}: without it every sample "
  783|                     f"would be decoded at the same tile grid forever, at a fixed offset from the "
  784|                     f"segment start, and no shape or count would differ. Pass a (B,) phase, or "
  785|                     f"decode densely with anchor_stride=1."
  786|                 )
  787|             phase = torch.zeros(batch, dtype=torch.long, device=device)
  788|         elif isinstance(anchor_phase, torch.Tensor):
  789|             phase = anchor_phase.to(dtype=torch.long).reshape(-1)
  790|             if phase.numel() != batch:
  791|                 raise ValueError(
  792|                     f"anchor_phase has {phase.numel()} entries but the batch is {batch}; the "
  793|                     f"phase is per sample, so a mismatch would tile one sample at another's grid"
  794|                 )
  797|             _refuse_phase_outside(phase, stride)
  798|             phase = phase.to(device=device, non_blocking=True)
  799|         else:
  800|             _refuse_phase_outside(torch.tensor([int(anchor_phase)]), stride)
  801|             phase = torch.full((batch,), int(anchor_phase), dtype=torch.long, device=device)
  802|
  803|         a_max = -(-span // stride)  # ceil, on ints
  804|         steps = torch.arange(a_max, device=device, dtype=torch.long)
  805|         anchors = floor + phase[:, None] + steps[None, :] * stride
  806|         valid = anchors < t_valid
  807|
  810|         count = (span - phase + stride - 1) // stride
  811|         last = floor + phase + (count - 1) * stride
  812|         return torch.where(valid, anchors, last[:, None]), valid
  813|
  814|     def forward(
  815|         self,
  816|         y_st: torch.Tensor,
  817|         y_ph: torch.Tensor,
  818|         u_stream: torch.Tensor,
  819|         anchor_phase: Optional[Union[int, torch.Tensor]] = None,
  820|         anchor_stride: Optional[int] = None,
  821|         source_valid: Optional[torch.Tensor] = None,
  822|     ) -> Dict[str, torch.Tensor]:
  875|         self._check_source_valid(source_valid, u_stream)
  876|         anchor_index, anchor_valid = self._build_anchor_index(
  877|             batch=int(y_st.shape[0]),
  878|             device=y_st.device,
  879|             anchor_phase=anchor_phase,
  880|             anchor_stride=anchor_stride,
  881|         )
  882|
  887|         target = torch.cat([y_st, y_ph], dim=-1)
  897|         persistence = (
  898|             self._anchor_target_values(target, anchor_index)
  899|             if self.persistence_residual
  900|             else None
  901|         )
  902|         if self.target_gate is not None:
  903|             target = self.target_gate(target)
  904|         source = u_stream if self.source_gate is None else self.source_gate(u_stream)
  905|
  906|         h_y = self.target_encoder(self.target_adapter(target))
  911|         h_u = self.encode_source_kv(source)
  912|
  921|         mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(
  922|             h_y,
  923|             clock=self._prior_clock(u_stream) if self.prior_availability_input else None,
  924|         )
  925|
  929|         query = (
  930|             torch.cat([mu_prior, logvar_prior], dim=-1)
  931|             if self.query_uses_logvar
  932|             else mu_prior
  933|         )
  934|         key_mask = self.build_lag_mask(h_u.shape[1], h_u.device)
  935|         source_open: Optional[torch.Tensor] = None
  936|         if source_valid is not None:
  937|             key_mask = self.source_key_mask(source_valid, key_mask, u_stream.dtype)
  938|             source_open = key_mask.any(dim=-1)
  939|         _, alpha, attended_heads = self.lag_attn(self.query_proj(query), h_u, key_mask)
  940|
  943|         event_gate = self.posterior_head.event_gate_values(attended_heads)
  944|         gate = event_gate
  945|         if source_open is not None:
  946|             gate, event_gate = self._close_gate(event_gate, source_open, attended_heads.dtype)
  947|         mu_post, logvar_post = self.posterior_head(
  948|             h_y, attended_heads, mu_prior, raw_logvar_prior, gate=gate
  949|         )
  950|         z_prior, z_post = self._reparameterize_shared(
  951|             mu_prior, logvar_prior, mu_post, logvar_post
  952|         )
  953|
  956|         with torch.no_grad():
  957|             mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
  958|             delta_mu_sat_frac = (
  959|                 (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
  960|             ).float().mean()
  961|
  965|         gather_index = anchor_index[:, :, None].expand(-1, -1, self.d_z)
  966|         mu_base, logvar_base = self.decoder(
  967|             z_prior.gather(1, gather_index), persistence=persistence
  968|         )
  969|         mu_full, logvar_full = self.decoder(
  970|             z_post.gather(1, gather_index), persistence=persistence
  971|         )
  972|
  976|         kld_btd = self.kld_tensor(
  977|             mu_prior=mu_prior,
  978|             logvar_prior=logvar_prior,
  979|             mu_post=mu_post,
  980|             logvar_post=logvar_post,
  981|         )
  982|         kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
  983|             kld_btd, alpha, head_structured=True
  984|         )
  985|
  986|         outputs = {
  987|             "mu_prior": mu_prior,
  988|             "logvar_prior": logvar_prior,
  989|             "raw_logvar_prior": raw_logvar_prior,
  990|             "mu_post": mu_post,
  991|             "logvar_post": logvar_post,
  992|             "z_prior": z_prior,
  993|             "z_post": z_post,
  994|             "target_state": h_y,
  995|             "source_state": h_u,
  996|             "attended_source_heads": attended_heads,
  997|             "attn_weights": alpha,
  998|             "mu_base": mu_base,
  999|             "logvar_base": logvar_base,
 1000|             "mu_full": mu_full,
 1001|             "logvar_full": logvar_full,
 1002|             "kld_per_t": kld_per_t,
 1003|             "kld_per_t_per_head": kld_per_t_per_head,
 1004|             "source_kl_lag_map": source_kl_lag_map,
 1005|             "mu_prior_sat_frac": mu_prior_sat_frac,
 1006|             "delta_mu_sat_frac": delta_mu_sat_frac,
 1007|             "anchor_index": anchor_index,
 1008|             "anchor_valid": anchor_valid,
 1009|         }
 1010|         if persistence is not None:
 1016|             outputs["persistence"] = persistence
 1017|         if event_gate is not None:
 1019|             outputs["event_gate"] = event_gate
 1020|         if source_open is not None:
 1022|             outputs["source_key_mask"] = key_mask
 1023|             outputs["source_open"] = source_open
 1024|         return outputs
 1025|
 1028|     def _check_source_valid(
 1029|         self, source_valid: Optional[torch.Tensor], u_stream: torch.Tensor
 1030|     ) -> None:
 1039|         enabled = getattr(self, "source_quality_mask", False)
 1040|         if source_valid is None:
 1041|             if enabled:
 1042|                 raise ValueError(
 1043|                     "source_quality_mask is on, so every forward needs source_valid: the source "
 1044|                     "steps the dataset's ua_step_mask marks usable. A caller that builds the "
 1045|                     "inputs itself must pass it as the sixth argument."
 1046|                 )
 1047|             return
 1048|         if not enabled:
 1049|             raise ValueError(
 1050|                 "source_valid was given to a model built without source_quality_mask. Build the "
 1051|                 "model with source_quality_mask=True to apply it."
 1052|             )
 1053|         batch, steps, width = (int(n) for n in u_stream.shape)
 1054|         shape = tuple(source_valid.shape)
 1055|         if shape not in ((batch, steps), (batch, steps, width)):
 1056|             raise ValueError(
 1057|                 f"source_valid has shape {shape}; expected ({batch}, {steps}) per step or "
 1058|                 f"({batch}, {steps}, {width}) per declared source channel."
 1059|             )
 1060|
 1061|     def source_key_mask(
 1062|         self,
 1063|         source_valid: torch.Tensor,
 1064|         lag_mask: torch.Tensor,
 1065|         dtype: torch.dtype = torch.float32,
 1066|     ) -> torch.Tensor:
 1094|         valid = source_valid.to(torch.bool)
 1095|         if valid.dim() == 3:
 1096|             if self.source_gate is not None:
 1097|                 valid = self.source_gate(valid.to(dtype)) > 0.5
 1098|             valid = valid.all(dim=-1)
 1099|         reach = self._source_kv_reach(int(valid.shape[1]))
 1100|         if reach > 1:
 1101|             bad = (~valid).to(dtype)[:, None, :]
 1103|             spread = F.max_pool1d(F.pad(bad, (reach - 1, 0)), kernel_size=reach, stride=1)
 1104|             valid = spread[:, 0, :] < 0.5
 1105|         num_lags = int(lag_mask.shape[-1])
 1106|         padded = torch.cat([valid.new_zeros(valid.shape[0], num_lags - 1), valid], dim=1)
 1108|         by_lag = padded.unfold(1, num_lags, 1).flip(-1)
 1109|         return lag_mask.to(torch.bool)[None] & by_lag
 1110|
 1111|     def _source_kv_reach(self, num_steps: int) -> int:
 1120|         body_of = getattr(self, "source_kv_body", None)
 1121|         body = body_of() if callable(body_of) else None
 1122|         if body is None:
 1123|             return 1
 1124|         reach = getattr(body, "receptive_field", 1)
 1125|         return int(num_steps) if reach is None else max(1, min(int(reach), int(num_steps)))
 1126|
 1127|     @staticmethod
 1128|     def _close_gate(
 1129|         event_gate: Optional[torch.Tensor], source_open: torch.Tensor, dtype: torch.dtype
 1130|     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
 1149|         open_steps = source_open[..., None].to(dtype)
 1150|         if event_gate is None:
 1151|             return open_steps, None
 1152|         effective = event_gate * open_steps
 1153|         return effective, effective
 1154|
 1156| __all__ = ["CAUSAL_ONLY_KEYWORDS", "FORWARDED_EXCLUSIONS", "CausalWarmupInputs"]
```

## 13. Causal feature-target mixin — `teb_vae/lag_attn_cfs/nets/causal_feature_target.py` (`CausalFeatureForecastTarget`)

Block splits (36/36), anchor-floor refusal, channel-weight resolution and buffer, likelihood structure (`_set_likelihood_structure` and `_register_likelihood_structure`: the `target_cell_mask` $(H,C_{\mathrm{keep}})$ from `target_scored_horizon`, and the `target_ar_logit` parameter under `forecast_ar_residual`), persistence admission and anchor-value gather, forecast-clock target gather, pooled validity, `compute_loss` override.

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
  440|     def _set_target_novelty(
  441|         self, *, target_novelty_frac: Optional[Sequence[float]]
  442|     ) -> None:
  463|         self.target_novelty_frac = (
  464|             None
  465|             if target_novelty_frac is None
  466|             else tuple(float(share) for share in target_novelty_frac)
  467|         )
  468|
  469|     def _set_channel_weights(self, *, target_weight_st: float, target_weight_ph: float) -> None:
  483|         self.target_weight_st = float(target_weight_st)
  484|         self.target_weight_ph = float(target_weight_ph)
  485|
  486|     def _register_channel_weights(self) -> None:
  499|         declared = (
  500|             torch.arange(self.c_y)
  501|             if self.target_gate is None
  502|             else self.target_gate.keep_index.cpu()
  503|         )
  504|         self.register_buffer(
  505|             "target_channel_weight",
  506|             self._resolve_channel_weights(
  507|                 declared.tolist(),
  508|                 weight_st=self.target_weight_st,
  509|                 weight_ph=self.target_weight_ph,
  510|             ),
  511|             persistent=False,
  512|         )
  513|
  514|     def _set_likelihood_structure(
  515|         self,
  516|         *,
  517|         target_scored_horizon: Optional[Sequence[int]],
  518|         forecast_ar_residual: bool,
  519|     ) -> None:
  535|         self.target_scored_horizon = (
  536|             None
  537|             if target_scored_horizon is None
  538|             else tuple(int(steps) for steps in target_scored_horizon)
  539|         )
  540|         self.forecast_ar_residual = bool(forecast_ar_residual)
  541|
  542|     def _register_likelihood_structure(self) -> None:
  557|         width = int(self.decoder_out_channels)
  558|         if self.target_scored_horizon is not None:
  559|             steps = self.target_scored_horizon
  560|             if len(steps) != int(self.c_y):
  561|                 raise ValueError(
  562|                     f"target_scored_horizon has {len(steps)} entries against c_y={int(self.c_y)}; "
  563|                     f"it is positional over the declared channels"
  564|                 )
  565|             if min(steps) < 1 or max(steps) > int(self.horizon):
  566|                 raise ValueError(
  567|                     f"target_scored_horizon entries must lie in [1, H={int(self.horizon)}], got "
  568|                     f"[{min(steps)}, {max(steps)}]"
  569|                 )
  570|             declared = (
  571|                 torch.arange(self.c_y)
  572|                 if self.target_gate is None
  573|                 else self.target_gate.keep_index.cpu()
  574|             )
  575|             kept = torch.as_tensor([steps[int(index)] for index in declared.tolist()])
  576|             taus = torch.arange(int(self.horizon))
  577|             self.register_buffer(
  578|                 "target_cell_mask",
  579|                 (taus[:, None] < kept[None, :]).to(torch.float32),
  580|                 persistent=False,
  581|             )
  582|         if self.forecast_ar_residual:
  583|             self.target_ar_logit = torch.nn.Parameter(torch.zeros(width))
  584|
  585|     @classmethod
  586|     def _resolve_channel_weights(
  587|         cls,
  588|         keep_index: Sequence[int],
  589|         *,
  590|         weight_st: float,
  591|         weight_ph: float,
  592|     ) -> torch.Tensor:
  626|         for name, value in (("target_weight_st", weight_st), ("target_weight_ph", weight_ph)):
  627|             if not float(value) >= 0.0:  # catches NaN as well as negatives
  628|                 raise ValueError(f"{name} must be >= 0 and not NaN, got {value!r}")
  629|
  630|         declared = torch.as_tensor(list(keep_index), dtype=torch.long)
  631|         weights = torch.where(
  632|             declared < cls.TARGET_BLOCK_SPLIT,
  633|             torch.tensor(float(weight_st)),
  634|             torch.tensor(float(weight_ph)),
  635|         ).to(torch.float32)
  636|
  637|         total = float(weights.sum())
  638|         if total <= 0.0:
  639|             raise ValueError(
  640|                 "target_weight_st and target_weight_ph are both zero, which would leave the "
  641|                 "objective with no reconstruction term; at least one block must carry weight"
  642|             )
  643|         return weights * (float(weights.numel()) / total)
  644|
  683|     def _check_persistence_target(self) -> None:
  697|
  698|     def _build_forecast_target(
  699|         self, target_features: torch.Tensor, anchors: Optional[torch.Tensor] = None
  700|     ) -> torch.Tensor:
  731|         shift = getattr(self, "target_forecast_shift", None)
  732|         if not shift or not any(shift):
  733|             return super()._build_forecast_target(target_features, anchors)
  734|
  737|         if target_features.dim() != 3:
  738|             raise ValueError(
  739|                 f"target stream must be 3-D (B, T, c_y), got shape {tuple(target_features.shape)}"
  740|             )
  741|         if target_features.size(1) != self.geometry.t:
  742|             raise ValueError(
  743|                 f"target stream length {target_features.size(1)} != geometry.t "
  744|                 f"{self.geometry.t}; this geometry assumes the trimmed loader "
  745|                 f"(trim_minutes: 1.0 -> T = {self.geometry.t} decimated steps), so a mismatch "
  746|                 f"means the loader ran at a different trim_minutes"
  747|             )
  748|         if target_features.size(2) != self.c_y:
  749|             raise ValueError(
  750|                 f"target stream has {target_features.size(2)} channels but the model declares "
  751|                 f"c_y={self.c_y}; the surviving-channel index is positional into the declared "
  752|                 f"width, so a mismatch would gather the wrong channels rather than fail"
  753|             )
  754|
  755|         gathered = (
  756|             target_features
  757|             if self.target_gate is None
  758|             else torch.index_select(target_features, -1, self.target_gate.keep_index)
  759|         )
  760|         if anchors is None:
  765|             anchors = (
  766|                 torch.arange(self.anchor_ceiling, device=gathered.device)
  767|                 .unsqueeze(0)
  768|                 .expand(gathered.shape[0], -1)
  769|             )
  770|
  771|         batch, channels = gathered.shape[0], gathered.shape[-1]
  772|         steps = torch.arange(self.horizon, device=gathered.device)
  775|         base_index = anchors.to(torch.long)[:, :, None] + 1 + steps[None, None, :]
  776|         out = gathered.new_empty(batch, anchors.shape[1], self.horizon, channels)
  777|         for value in sorted(set(shift)):
  778|             columns = torch.tensor(
  779|                 [index for index, s in enumerate(shift) if s == value],
  780|                 dtype=torch.long,
  781|                 device=gathered.device,
  782|             )
  783|             block = torch.index_select(gathered, -1, columns)
  784|             time_index = (
  785|                 (base_index + int(value))
  786|                 .reshape(batch, -1, 1)
  787|                 .expand(-1, -1, columns.numel())
  788|             )
  789|             window = block.gather(1, time_index).reshape(
  790|                 batch, anchors.shape[1], self.horizon, columns.numel()
  791|             )
  792|             out.index_copy_(-1, columns, window)
  793|         return out
  794|
  795|     def scored_weight(self, weight: torch.Tensor) -> torch.Tensor:
  826|         return pooled_scored_weight(
  827|             weight, getattr(self, "target_forecast_shift", None)
  828|         )
  829|
  830|     def compute_loss(
  831|         self,
  832|         forward_outputs: Dict[str, torch.Tensor],
  833|         target_features: torch.Tensor,
  834|         *,
  835|         weight: torch.Tensor,
  836|         beta: float = 1.0,
  837|         beta_prior: float = 0.0,
  838|         lambda_full: float = 1.0,
  839|         lambda_base: float = 1.0,
  840|         likelihood: str = "gaussian_nll",
  841|         free_bits: float = 0.0,
  842|         lambda_ms: float = 0.0,
  843|         lambda_deriv: float = 0.0,
  844|         lambda_boundary: float = 0.0,
  845|         lambda_event_gate: float = 0.0,
  846|         full_anchor_weight: Optional[torch.Tensor] = None,
  847|     ) -> Dict[str, Any]:
  863|         return super().compute_loss(
  864|             forward_outputs,
  865|             target_features,
  866|             weight=self.scored_weight(weight),
  867|             beta=beta,
  868|             beta_prior=beta_prior,
  869|             lambda_full=lambda_full,
  870|             lambda_base=lambda_base,
  871|             likelihood=likelihood,
  872|             free_bits=free_bits,
  873|             lambda_ms=lambda_ms,
  874|             lambda_deriv=lambda_deriv,
  875|             lambda_boundary=lambda_boundary,
  876|             lambda_event_gate=lambda_event_gate,
  877|             full_anchor_weight=full_anchor_weight,
  878|         )
  879|
  880|     def _anchor_target_values(
  881|         self, target_features: torch.Tensor, anchors: torch.Tensor
  882|     ) -> torch.Tensor:
  921|         gathered = (
  922|             target_features
  923|             if self.target_gate is None
  924|             else torch.index_select(target_features, -1, self.target_gate.keep_index)
  925|         )
  926|         shift = getattr(self, "target_forecast_shift", None)
  927|         if not shift or min(shift) >= 0:
  928|             index = anchors.to(torch.long)[:, :, None].expand(-1, -1, gathered.shape[-1])
  929|         else:
  932|             offsets = torch.tensor(
  933|                 [min(int(s), 0) for s in shift], dtype=torch.long, device=gathered.device
  934|             )
  935|             index = anchors.to(torch.long)[:, :, None] + offsets[None, None, :]
  936|         values = gathered.gather(1, index)
  940|         waits = getattr(self, "target_warm_steps", None)
  941|         if waits is None:
  942|             return values
  943|         return values * (index >= waits).to(values.dtype)
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
   89|         posterior_event_gate: bool = False,
   90|         lag_kv_source: str = "encoder",
   91|         use_entmax: bool = False,
   92|         attention_grad_checkpoint: bool = False,
   93|         lag_bias_init: str = "normal",
   94|         alibi_slope_scale: float = 1.0,
   95|         query_uses_logvar: bool = False,
   96|         prior_availability_input: bool = False,
   97|         causal_norm: bool = False,
   98|         coverage_floor: float = 0.9,
   99|         base_decode: str = "sample",
  100|         persistence_residual: bool = False,
  101|         horizon_weight_halflife_steps: Optional[float] = None,
  102|         target_keep_index: Optional[Sequence[int]] = None,
  103|         target_warmup_steps: Optional[Sequence[int]] = None,
  104|         source_keep_index: Optional[Sequence[int]] = None,
  105|         source_warmup_steps: Optional[Sequence[int]] = None,
  106|         target_align_delays: Optional[Sequence[int]] = None,
  107|         source_align_delays: Optional[Sequence[int]] = None,
  108|         anchor_stride: int = 1,
  109|         lag_floor: int = 0,
  110|         target_weight_st: float = 1.0,
  111|         target_weight_ph: float = 1.0,
  112|         target_novelty_frac: Optional[Sequence[float]] = None,
  113|         target_forecast_shift: Optional[Sequence[int]] = None,
  114|         target_scored_horizon: Optional[Sequence[int]] = None,
  115|         forecast_ar_residual: bool = False,
  116|         source_quality_mask: bool = False,
  117|         init_weights: bool = True,
  118|     ) -> None:
  188|         forwarded = {
  189|             name: value
  190|             for name, value in locals().items()
  191|             if name not in FORWARDED_EXCLUSIONS
  192|         }
  193|
  195|         self._set_causal_inputs(
  196|             horizon=horizon,
  197|             target_keep_index=target_keep_index,
  198|             target_warmup_steps=target_warmup_steps,
  199|             source_keep_index=source_keep_index,
  200|             source_warmup_steps=source_warmup_steps,
  201|             anchor_stride=anchor_stride,
  202|             lag_floor=lag_floor,
  203|             target_forecast_shift=target_forecast_shift,
  204|             source_quality_mask=source_quality_mask,
  205|         )
  208|         self._set_channel_weights(
  209|             target_weight_st=target_weight_st, target_weight_ph=target_weight_ph
  210|         )
  214|         self._set_target_novelty(target_novelty_frac=target_novelty_frac)
  217|         self._set_likelihood_structure(
  218|             target_scored_horizon=target_scored_horizon,
  219|             forecast_ar_residual=forecast_ar_residual,
  220|         )
  221|
  227|         super().__init__(
  228|             **forwarded,
  229|             target_delays=target_align_delays,
  230|             source_delays=source_align_delays,
  231|         )
  232|
  234|         self._validate_causal_geometry()
  237|         self._register_channel_weights()
  239|         self._register_likelihood_structure()
```

### 14.2 `SeqVaeLagAttnTrfCfs` — `teb_vae/lag_attn_transformer_cfs/nets/model.py`

`teb_vae/lag_attn_transformer_cfs/nets/model.py` (docstrings/comments stripped; `NNNN|` = original line)

```python
   48| from __future__ import annotations
   49|
   50| from typing import Optional, Sequence, Tuple
   51|
   52| from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
   53| from teb_vae.lag_attn_cfs.nets.causal_inputs import (
   54|     FORWARDED_EXCLUSIONS,
   55|     CausalWarmupInputs,
   56| )
   57| from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
   58|
   60| class SeqVaeLagAttnTrfCfs(
   61|     CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws
   62| ):
   78|
   79|     def __init__(
   80|         self,
   81|         *,
   82|         sequence_length: int = 300,
   83|         d_model: int = 128,
   84|         d_z: int = 48,
   85|         horizon: int = 30,
   86|         raw_per_step: int = 16,
   87|         warmup_period: int = 134,
   88|         c_y: int = 102,
   89|         c_u: int = 51,
   90|         use_up_st: bool = True,
   91|         max_lag: int = 90,
   92|         num_heads: int = 4,
   93|         d_head: int = 32,
   94|         dropout: float = 0.1,
   95|         decoder_hidden: int = 128,
   96|         horizon_depth: int = 2,
   97|         horizon_kernel: int = 3,
   98|         horizon_film: bool = False,
   99|         horizon_attention_blocks: int = 0,
  100|         horizon_embed_std: float = 0.02,
  101|         head_init_calibration: bool = False,
  102|         a_head_gain: float = 1.0,
  103|         encoder_conv_kernels: Sequence[int] = (5, 9),
  104|         encoder_conv_dilations: Sequence[int] = (1, 2),
  105|         encoder_num_heads: int = 4,
  106|         encoder_d_ff: int = 256,
  107|         target_attention_blocks: int = 4,
  108|         source_attention_blocks: int = 3,
  109|         source_attention_window: Optional[int] = 16,
  110|         logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
  111|         mu_scale: float = 5.0,
  112|         delta_mu_scale: float = 3.0,
  113|         delta_logvar_scale: float = 2.0,
  114|         posterior_logvar_mode: str = "residual",
  115|         source_dropout: Optional[float] = None,
  116|         posterior_event_gate: bool = False,
  117|         lag_kv_source: str = "encoder",
  118|         use_entmax: bool = False,
  119|         attention_grad_checkpoint: bool = False,
  120|         lag_bias_init: str = "normal",
  121|         alibi_slope_scale: float = 1.0,
  122|         query_uses_logvar: bool = False,
  123|         prior_availability_input: bool = False,
  124|         coverage_floor: float = 0.9,
  125|         base_decode: str = "sample",
  126|         persistence_residual: bool = False,
  127|         horizon_weight_halflife_steps: Optional[float] = None,
  128|         target_keep_index: Optional[Sequence[int]] = None,
  129|         target_warmup_steps: Optional[Sequence[int]] = None,
  130|         source_keep_index: Optional[Sequence[int]] = None,
  131|         source_warmup_steps: Optional[Sequence[int]] = None,
  132|         target_align_delays: Optional[Sequence[int]] = None,
  133|         source_align_delays: Optional[Sequence[int]] = None,
  134|         anchor_stride: int = 1,
  135|         lag_floor: int = 0,
  136|         target_weight_st: float = 1.0,
  137|         target_weight_ph: float = 1.0,
  138|         target_novelty_frac: Optional[Sequence[float]] = None,
  139|         target_forecast_shift: Optional[Sequence[int]] = None,
  140|         target_scored_horizon: Optional[Sequence[int]] = None,
  141|         forecast_ar_residual: bool = False,
  142|         source_quality_mask: bool = False,
  143|         init_weights: bool = True,
  144|     ) -> None:
  218|         forwarded = {
  219|             name: value
  220|             for name, value in locals().items()
  221|             if name not in FORWARDED_EXCLUSIONS
  222|         }
  223|
  225|         self._set_causal_inputs(
  226|             horizon=horizon,
  227|             target_keep_index=target_keep_index,
  228|             target_warmup_steps=target_warmup_steps,
  229|             source_keep_index=source_keep_index,
  230|             source_warmup_steps=source_warmup_steps,
  231|             anchor_stride=anchor_stride,
  232|             lag_floor=lag_floor,
  233|             target_forecast_shift=target_forecast_shift,
  234|             source_quality_mask=source_quality_mask,
  235|         )
  238|         self._set_channel_weights(
  239|             target_weight_st=target_weight_st, target_weight_ph=target_weight_ph
  240|         )
  244|         self._set_target_novelty(target_novelty_frac=target_novelty_frac)
  247|         self._set_likelihood_structure(
  248|             target_scored_horizon=target_scored_horizon,
  249|             forecast_ar_residual=forecast_ar_residual,
  250|         )
  251|
  257|         super().__init__(
  258|             **forwarded,
  259|             target_delays=target_align_delays,
  260|             source_delays=source_align_delays,
  261|         )
  262|
  264|         self._validate_causal_geometry()
  267|         self._register_channel_weights()
  269|         self._register_likelihood_structure()
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
  401|         lambda_event_gate: float = 0.0,
  402|         full_anchor_weight: Optional[torch.Tensor] = None,
  403|     ) -> Dict[str, Any]:
  460|         anchors: Optional[torch.Tensor] = forward_outputs.get("anchor_index")
  461|         target = (
  462|             build_future_target(fhr_raw, self.geometry, future_index=self.future_index)
  463|             if anchors is None
  464|             else gather_anchored_future_target(
  465|                 fhr_raw, self.geometry, anchors, future_index=self.future_index
  466|             )
  467|         )
  468|         result = compute_raw_objective(
  469|             forward_outputs,
  470|             target,
  471|             weight=weight,
  472|             geometry=self.geometry,
  474|             block_width=self.geometry.r,
  475|             coverage_floor=self.coverage_floor,
  476|             logvar_clamp=self.logvar_clamp,
  477|             beta=beta,
  478|             beta_prior=beta_prior,
  479|             lambda_full=lambda_full,
  480|             lambda_base=lambda_base,
  481|             likelihood=likelihood,
  482|             free_bits=free_bits,
  483|             lambda_ms=lambda_ms,
  484|             lambda_deriv=lambda_deriv,
  485|             lambda_boundary=lambda_boundary,
  486|             lambda_event_gate=lambda_event_gate,
  487|             full_anchor_weight=full_anchor_weight,
  492|             horizon_weight=getattr(self, "horizon_weight", None),
  493|         )
  496|         result["metrics"]["anchors_per_sample"] = self._anchors_per_sample(
  497|             forward_outputs, target
  498|         )
  499|         result["metrics"].update(self._source_lag_warmth(forward_outputs, target))
  500|         return result
  501|
  503| __all__ = ["CausalRawInputs", "gather_anchored_future_target"]
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
