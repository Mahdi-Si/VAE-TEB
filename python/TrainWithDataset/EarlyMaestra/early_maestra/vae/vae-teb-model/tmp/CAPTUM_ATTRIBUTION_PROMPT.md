# Prompt: Captum-based mechanistic analysis of the two causal-feature forecasters

## Your task, in one paragraph

Build a Captum attribution analysis for the two models `teb_vae/lag_attn_transformer_cfs` (lag
*attention*) and `teb_vae/lag_slot_transformer_cfs` (lag *residual/slot*), attach it to each
model's evaluation pipeline as a first-class, documented, tested analysis, produce advanced
visualisations of what happens inside each model — which input coefficients, at which stored-time
offsets and which frequency bands, drive the latent, the divergence $K_t$ and the forecast gain —
and write the design-and-findings document for it. Work in three phases, in order, and do not start
a later phase before the earlier one is genuinely understood: **(1) understand the models and their
eval pipelines, (2) understand Captum against these models' constraints, (3) implement, test,
document.** Never commit. Use the project interpreter `.venv/Scripts/python.exe`. Read `CLAUDE.md`
(repo root) first and obey it throughout; the rules below restate the ones most likely to bite.

## Phase 1 — understand the models and the pipelines (read before you write anything)

Start from what already exists and from the fact that a per-recording trace analysis was just
added — it is the closest precedent for what you will build, and much of its plumbing is yours to
reuse:

* `teb_vae/lag_attn_cfs/eval/analyses/recording_traces.py` (the lag-attentive cells' analysis;
  registered in `lag_attn_cfs/eval/run.py::ANALYSIS_FUNCTIONS` and therefore run by **both**
  `lag_attn_cfs` and `lag_attn_transformer_cfs`), `teb_vae/lag_attn_cfs/eval/traces.py` (the
  torch-free shared core: selection, reductions, figures), `teb_vae/lag_attn_cfs/eval/dataset_rows.py`
  (row → dataset-index mapping, sequential `subset_loader`, `check_batch_identity` — **use these; do
  not re-implement them**), `teb_vae/lag_slot_transformer_cfs/eval/recording_traces.py` (the slot
  cell's post-pass stage, called from its `run.py::main`), and the tests
  `tests/test_eval_recording_traces.py` in both cells. These show exactly how an analysis that must
  touch the model selects segments, re-reads them through the loader with an identity check, runs a
  dense forward (`anchor_phase=0, anchor_stride=1`; the slot cell adds `return_proposals=True`), and
  writes files, figures and a summary block under the documentation contracts.
* Read, in this order: `teb_vae/lag_attn_cfs/eval/EVAL.md` (the whole contract: layering, protocol,
  output layout, every analysis, "how the output is misread"), `teb_vae/lag_attn_cfs/eval/FIGURE_GUIDE.md`,
  `teb_vae/lag_attn_cfs/eval/analyses/__init__.py` (the analysis protocol), `run.py`, `metrics.py`
  (`evaluate_batch`, `anchor_support`, `model_inputs`, `DENSE_ANCHOR_GEOMETRY`), `collect.py` (the
  durable tables: `per_sample.csv`, `per_anchor.parquet`, `per_anchor_vectors.npz`), `lag_axis.py`
  (the lag axis is **stored-coefficient time**, never physiological delay — the caveat and its
  constants), `lag_shape.py`, `figures_seam.py`, `analyses/samples.py`, `analyses/occlusion.py` (the
  existing *interventional* readout — your attributions will be compared against it),
  `analyses/source_null.py` (the availability-clock hazard: the source availability pattern alone
  can move the posterior; any attribution of $K_t$ to the source must be read against this),
  `analyses/band_partition.py` (the channel → frequency-band map; `band_channel_map_kept.csv` is the
  join you will use for band-resolved attributions).
* Then the model side: `teb_vae/lag_attn_transformer_cfs/DESIGN.md`, `nets/model.py` (three-way
  MRO), `teb_vae/lag_attn_cfs/nets/causal_inputs.py` (the forward, the input warm-up gate, lag
  floor, anchor tiling), `teb_vae/lag_attn_cfs/nets/causal_feature_target.py`,
  `teb_vae/lag_attn_transformer_rws/nets/model.py` (encoders, prior/posterior heads, lag
  cross-attention with `entmax15`, head-structured posterior), `teb_vae/lag_attn/nets/heads.py::TEAnalysisHead`.
  For the slot cell: `teb_vae/lag_slot_transformer_cfs/DESIGN.md`, `MODEL_EXPLAINED.md`,
  `nets/model.py`, `nets/core.py`, `nets/lag_updates.py`, `nets/lag_attention.py`,
  `nets/controls.py::suppressed_parameters`, `eval/run.py` (`score_batch`, `run_pass`, `main`),
  `eval/binding.py` (`ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE`), `eval/verify.py::FORBIDDEN_KEYS`
  (attention-shaped key names are refused anywhere in that cell's summary — your names must not
  collide).
* Dataset facts you must hold: inputs are one-sided scattering (`*_st`) and phase-harmonic (`*_ph`)
  coefficient blocks on a decimated grid, z-scored per channel, with per-channel causal warm-up
  (`causal_warmup_steps`) and composed group delay (`causal_delay_s`) stored as shard attributes;
  the model sees no raw signal; `epoch` is seconds before delivery; class comes from
  `target/weight`; subgroup from the shard basename. **The stored UP/FHR timeline is canonical: no
  mechanical-shift, sensor-delay, `up_shift_secs` or `tau_pre` term anywhere, in code, plots,
  captions or docs** (this is enforced by tests in the slot cell and is a standing repo rule).
* Write down, before Phase 2, a one-page summary in your own words of: the 22-key forward contract
  of each cell, the anchor axis vs the time axis (dense $T$ latents vs gathered anchors in the
  attention cell; everything already on the anchor axis in the slot cell), what "lag" means in each,
  which quantities are per-anchor scalars you can attribute (e.g. $K_t$, `pred_gap` per anchor, a
  latent coordinate $\mu^q_{t,d}$, the attention weight at a lag, a proposal norm), and which
  tensors are non-differentiable or masked.

## Phase 2 — understand Captum against these constraints

Check `requirements.txt` and the lockfile for Captum; if absent, install it into `.venv` and add it
to `requirements.txt` with a one-line justification (per `CLAUDE.md`'s dependency rule — a
gradient-attribution library is not a few lines to own locally). Then verify, on the **tiny models
the test suites build** (`teb_vae/lag_attn_cfs/tests/conftest.py::make_task`/`make_stub_batch`;
`teb_vae/lag_slot_transformer_cfs/tests/conftest.py::build_tiny_model`/`tiny_streams`), which
Captum methods actually run on these forwards, and record what fails and why:

* Write a thin `nn.Module` wrapper whose `forward(y_st, y_ph, u_stream)` calls the real model
  densely and returns **one scalar per sample** for a chosen anchor and a chosen readout. Readouts
  to support at minimum: $K_t$ (`kld_per_t`/`kld_per_anchor` at the anchor), the per-coordinate
  divergence and $\mu^q_{t,d}$ for a chosen latent coordinate, the per-anchor block NLL of the full
  and base branches and their gap, and — attention cell only — the attention mass on a chosen lag
  band; slot cell only — the proposal norm on a lag band. Decide and document whether the sampled
  branch (`z_post`) or the mean is attributed (means, unless the readout is defined on the sample;
  seed and freeze $\epsilon$ if it is).
* Evaluate at least: `IntegratedGradients`, `GradientShap`, `Saliency`/`InputXGradient`,
  `FeatureAblation`/`Occlusion` (model-agnostic; relate to the existing `occlusion` analysis),
  `LayerIntegratedGradients`/`LayerConductance` on the latent heads and on the attended per-head
  summaries, `NeuronConductance` on the most active latent coordinates. Test `DeepLift` and note
  that custom modules (entmax, RMSNorm, the residual limiter) may not be supported — do not force
  it. Check completeness (IG's attribution sum vs the output delta) and convergence (`n_steps`,
  `return_convergence_delta`).
* Baselines are a modelling decision, not a default: under z-scoring a zero coefficient is the
  channel mean (the climatology baseline the pipeline already uses), and the warm-up gate already
  zeroes unavailable steps. Design and justify at least two baselines: all-zero inputs, and
  **source-only-zeroed with the target stream held fixed** — the exact null arm `source_null` uses
  — so that attributions of $K_t$ or `pred_gap` to the source are attributions to source *content*
  rather than to the availability clock. State how each baseline interacts with the warm-up gate.
* Structural checks you must be able to assert: (a) **causality** — attribution to any input step
  later than the anchor is exactly zero; (b) attribution to gated-off (not-yet-warm) channels is
  zero; (c) the source attribution of a target-only readout ($\mu^p$, base NLL) is zero; (d) IG
  completeness within tolerance. These become tests.

## Phase 3 — implement

**Where and how (follow the precedent exactly):**

* Shared core: `teb_vae/lag_attn_cfs/eval/attributions.py` (layer 1; torch and Captum allowed) —
  the wrapper module, the readout registry, baseline construction, the batched attribution loop,
  and reductions: per-anchor attribution maps over (stored time × channel) for each stream; the
  **lag-aligned profile** (attribution re-indexed by offset $\ell = t_{\text{anchor}} - t$ so it can
  be laid beside the model's own attention / proposal profile on the same compensated seconds axis
  from `lag_axis.compensated_seconds_axis`); the **band-resolved** attribution through
  `band_channel_map_kept.csv`; the **agreement** between attribution and the model's own lag readout
  (per-anchor correlation / Jensen–Shannon against `source_kl_lag_map` and the head-averaged
  attention in the attention cell, against the proposal norm in the slot cell); and per-class /
  per-subgroup summaries on per-recording units, never per anchor. Keep the figure builders
  torch-free and in the style of `traces.py`/`figures_seam.py`.
* Attention cell: `teb_vae/lag_attn_cfs/eval/analyses/attribution.py`, registered once in
  `run.py::ANALYSIS_FUNCTIONS` (it then runs for `lag_attn_transformer_cfs` too — do **not**
  register anything locally there). It reads `context.task`/`context.loader` (add it to
  `MODEL_READING_ANALYSES` in `tests/test_eval_protocol.py`), records a skip without a model,
  selects segments with a seeded class-balanced draw (reuse `traces.select_recordings` at recording
  level or `samples.per_class_rows` at segment level — say which and why), re-reads them via
  `dataset_rows`, and joins `per_anchor.parquet` where useful. Add a `caps.attribution_segments`-style
  cap to **both** cfs cells' `eval/configs/eval_overrides.yaml` (their `eval_config` blocks are
  asserted identical) and document it there and in `EVAL.md`.
* Slot cell: `teb_vae/lag_slot_transformer_cfs/eval/attribution.py` as a post-pass stage called
  from `run.py::main` inside a failure-isolating guard (model on `recording_traces` / `run_traces`),
  reusing the identities the pass now records; its block goes into `summary.json` under a name free
  of `FORBIDDEN_KEYS`.
* Advanced analysis to include (each with a figure and a table, each on per-recording units with
  counts): (1) per-anchor attribution heatmaps for both streams with the anchor marked and the
  warm-up boundary drawn; (2) the lag-aligned source attribution profile vs the model's own lag
  readout, pooled and by class, with the agreement statistic; (3) band-resolved attribution of
  `pred_gap` and $K_t$ (which frequency bands of the source informed the forecast), compared side by
  side with `occlusion`'s per-band delta and `spectral_skill`; (4) latent-layer attributions: which
  input regions write which latent coordinates / heads, and how the top-KL coordinates are fed;
  (5) **attribution along a recording's trace** — for a few recordings the trace analysis already
  selected, how the lag-aligned attribution moves over hours before delivery (reuse the trace's
  file layout and figure primitives); (6) the null-baseline decomposition: attribution under the
  source-zeroed baseline vs the all-zero baseline, stated as content vs clock. Every lag axis is
  stored-coefficient time with the caveat printed on the figure; if you map attributions from
  coefficient time toward raw time using the shard's per-channel `causal_delay_s`, do it as a
  clearly labelled *secondary* view and never introduce a source-timeline shift.
* Cost control: attribution is per anchor and per readout, so cap segments and sample anchors (e.g.
  a seeded subset per segment, or every $k$-th anchor), batch over anchors where the wrapper allows,
  run under `torch.no_grad()` only where gradients are not needed, and record the measured cost in
  the block as `occlusion` does. Never run whole test packages "to be safe".

**Conventions the tests enforce (they will fail otherwise):** analysis signature
`run_<name>_analysis(context, *, eval_config, output_dir, probe)` returning `n_samples`,
`composition`, `plan{capped…}`; no analysis imports another (shared code moves down a layer);
layering rules in `tests/test_eval_self_contained.py`; a `### <name>` section in
`lag_attn_cfs/eval/EVAL.md` (and **none** in the transformer cell's `EVAL.md`); a row in `run.py`'s
`RUN_ARGS` comment table; `FIGURE_GUIDE.md` entries for every fixed figure and a family entry for
dynamically named ones; `eval/figure_manifest.json` plus `FAMILIES`/`observed_figures` in
`tests/test_eval_smoke.py`; new own-only modules declared in `CELL_SPECIFIC_MODULES` of
`tests/test_eval_divergences.py`; `cohort.py` must stay byte-identical to the `lag_attn_rws`
sibling's (put shared helpers in `lag_axis.py`/`frames.py`, not there); the slot cell's
`tests/test_docs.py` (no `.md` references in code, no geometry literals in prose, no
timeline-correction terms, a docstring on every public *and dunder* function including tests);
Google-style docstrings with LaTeX (`$…$`) for maths; any figure or manifest naming a GUID also
names its subgroup; no hard-coded horizon/anchor/channel numbers in comments — read them off the
model; never put an apostrophe inside a Bash heredoc (write scripts to a file and run them).

**Verification:** fast tests on the tiny models for the structural checks (causality, gating,
completeness, target-only zero source attribution, reductions, skip without model), one end-to-end
fast test through a stub loader, and a `@pytest.mark.slow` assertion on the `collected_run` /
`evaluated` fixtures. Run file-level selections only; announce any command expected to exceed a
minute and run it in the background; the slow suites are long on Windows, so hand those to the
user to run with `!` rather than launching them yourself. Note that
`tests/test_docs.py::test_no_module_references_a_markdown_document` in the slot cell,
`test_sweep_configs` (`sweep_legacy_dualref_physclock.yaml`) in the transformer cell and
`test_eval_metrics.py::test_zero_is_the_channel_mean_over_the_region_the_model_reads` in the cfs
cell are already red for reasons unrelated to this work.

**Document:** write `teb_vae/lag_attn_cfs/eval/ATTRIBUTION.md` (and a short pointer paragraph in
the slot cell's `EVAL.md`), in the house style of `EVAL.md`/`LATENT_TRAJECTORY.md`: what is
attributed to what, the wrapper and readouts, the baselines and why, which Captum methods work on
these forwards and which do not and why, the structural checks and what they prove, every table and
figure with its axes and units, the cost, and a full **"how this output will be misread"** section
— attribution is not causation, the lag axis is coefficient time, the availability clock, the OOD
classes, per-recording units. Include the findings from the tiny/fixture runs you actually made,
labelled as fixture findings, and leave the production-run findings as a clearly marked section for
the user to fill from a real checkpoint.

**Report at the end:** what was implemented and where, exactly which tests you ran with their
results, which methods Captum could not run on these models and why, the commands the user should
run for the slow suites, and anything you deliberately scoped out with a `lean-limit:` comment in
the code.
