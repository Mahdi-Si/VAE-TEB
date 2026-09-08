# Minimal pilot: outcome separation in CFS Transformer latents near delivery

Written 2026-09-07. **Status: research and proposed protocol; no fine-tuning or outcome analysis has been run for this document.**

**Implementation/execution boundary (user instruction):** the coding agent writes all code, configuration, tests and documentation first, with only a minimal local check of isolated logic on synthetic inputs after the code is written. The actual experiment runs on another machine that the agents cannot access: the user runs integration tests, smoke checks, preflight, extraction, training, evaluation and report generation there through the editable `RUN_ARGS` dictionary and `__main__` entry point in `latent_pilot/run.py`, with CLI support as an alternative. Sections 1–10 describe the behavior to implement; §11 tracks coding work and its minimal logic check, and §12 specifies the user's separate execution work.

## 1. Recommendation and question

Use one existing classification fold, one pretrained checkpoint, and one seed. First fit a linear classifier on frozen latent summaries. Then fine-tune only the model's small posterior-mean output layers, together with the same kind of classifier. Supervise one **final-hour summary per recording**, preserve the pretrained representation across the last three hours, and evaluate separation and trajectories on recordings withheld from fitting and selection.

The question is:

> Can a small supervised change to the pretrained latent mean improve held-out separation of healthy versus adverse-outcome recordings near delivery, while retaining useful variation over the preceding three hours?

This tests outcome-associated representation learning. It does not establish the time of injury, an instantaneous fetal-health label, or a physiological progression score. Successful separation along a learned direction is already a useful preliminary result; two perfectly isolated clouds in an unsupervised plot are not required.

**Minimum experiment:** frozen baseline + one fine-tuned model, at most ten fine-tuning epochs, final-hour recording-level discrimination, six half-hour trajectory bins, three main figures, and basic forecast/latent-preservation checks. No architecture redesign, full cross-validation, onset detector, or large loss sweep is needed.

## 2. What the existing model actually learns

The starting reference is [CFS reference, §1](../CFS_CRWS_MODELS_REFERENCE.md). Executable code and the checkpoint's saved configuration take precedence over historical prose.

| Component | Relevant behavior and implication |
|---|---|
| Inputs | Causal FHR scattering/phase features and a separate UP scattering/phase stream. Keep the checkpoint's preprocessing, selected channels, normalization, warm-up and alignment. |
| Target branch | Causal convolution/Transformer encoding of FHR history produces the target-only latent distribution, returned as `mu_prior`, `logvar_prior`. |
| Source branch | Lag attention reads UP history. Its per-head summaries and FHR state enter the posterior head, returning `mu_post`, `logvar_post`. “Posterior” here does not mean that future forecast labels are encoder inputs. |
| Latent means | $\mu^q = \mu^p + \Delta\mu$. Both are $(B, T, d_z)$; the current YAML has `d_z: 64`. Use means for deterministic representation analysis, not sampled `z_post`. |
| Forecast | A shared decoder predicts future FHR-derived coefficients from each latent branch. A target-only persistence path also exists. The pretraining objective encourages prediction and regularizes the two distributions; it does not directly optimize clinical-class separation. |
| Interpretation | The posterior is a combined FHR+UP representation. Its residual `mu_post - mu_prior` is a useful secondary readout, but neither it nor the source-conditioned KL identifies disease severity or causal UP influence. |

Implementation anchors:

- [CFS Transformer composition](nets/model.py): `SeqVaeLagAttnTrfCfs`.
- [Causal forward path](../lag_attn_cfs/nets/causal_inputs.py): `CausalWarmupInputs.forward`, returned means, anchor indices and validity.
- [Transformer encoders](../lag_attn_transformer_rws/nets/encoders.py): causal history encoding.
- [Posterior head](../lag_attn/nets/heads.py): `PosteriorHead`, especially `fusion` and `delta_mu_head`.
- [Shared training losses](../lag_attn_rws/nets/losses.py): forecast scores and latent KL.

**A concrete reference drift:** the current [default YAML](configs/default.yaml) sets `horizon: 10` (40 seconds), whereas substantial portions of the reference describe $H=30$ (120 seconds). Representation versions also differ: the promoted integer-phase configuration and older ratio-power configurations have different channel contracts. Do not rebuild a pretrained model from today's defaults. Load its recorded `model_class`, `model_kwargs`, `resolved_config.yaml`, and matching statistics/shards strictly. This pilot is not a preprocessing migration.

No production checkpoint or classification dataset was identified in the repository file inventory inspected for this document. Checkpoint identity, training provenance, fold counts, and data availability remain execution-time inputs; no performance or sample-size claim is made here.

## 3. Clinical labels are weak temporal supervision

The [dataset builder](../../hdf5_dataset/new_pipeline/create_new_pipeline.py), `SUBGROUP_META`, defines:

| Stored class | Binary outcome for this pilot | Retain for reporting |
|---|---|---|
| `1`: healthy | `0` | Blood-gas/no-blood-gas and CS/no-CS subgroup |
| `2`: acidosis | `1` | Acidosis as a separate descriptive subgroup |
| `3`: HIE | `1` | HIE as a separate descriptive subgroup |

These codes are categories, not an ordinal severity scale. In particular, do not fit a regression target $1 < 2 < 3$, or assume every HIE case is a more severe version of every acidosis case. Use “adverse-outcome group” when discussing the pooled positive class.

The HDF5 `target` is a class code multiplied by validity weight. A zero can mean invalid data. Obtain the recording label from verified metadata or consistent positive class codes on valid samples; do not average `target` into a class or treat zeros as healthy. The existing helper `clinical_class_code` in [labels.py](../lag_attn/eval/labels.py) is a reuse point, with an additional explicit recording-level consistency check.

Let $y_i$ be a recording's observed outcome and $s_i(t)$ its unobserved physiological state. The data give $y_i$, not $s_i(t)$. A later HIE outcome does not supply an abnormal-state label for every earlier anchor. ACOG/AAP describe heterogeneous pathways and emphasize that attribution and timing require broader clinical evidence. That supports caution about temporal labels; it does not validate any particular one-hour boundary. [ACOG/AAP report](https://publications.aap.org/pediatrics/article/133/5/e1482/32738/Neonatal-Encephalopathy-and-Neurologic-Outcome).

Practical consequences:

- Apply the outcome loss to a late recording **bag/summary**, not a separate hard label at every four-second anchor across three hours.
- Give anchors one to three hours before delivery preservation supervision only. Do not relabel them healthy either.
- Do not impose a monotonic increase, prescribe an onset, or fabricate soft severity labels from time-to-delivery. Both gradual and abrupt changes, stable courses, and reversals must remain possible.
- Use delivery time to select and describe retrospective windows; do not feed it, CS status, blood-gas status, or outcome metadata to the classifier or encoder.

Bag-level supervision is motivated by multiple-instance learning, which associates labels with collections rather than requiring labels for every member. For this pilot, use fixed pooling; the learned attention aggregator of the cited paper is unnecessary. Bag-level training still distributes gradients across its members and cannot identify which moment was abnormal. [Ilse et al., ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html).

## 4. One-fold data protocol and the actual time axis

### 4.1 Split and cohort

Choose `fold_1` before looking at class-separation results. Use its existing train/validation/test assignment and record whether the dataset was built in `holdout` or `augmented` mode; the builder supports both. Never randomly split segments or anchors. Assert disjoint GUID sets across all splits and group related recordings by patient/delivery if a stronger identifier is available. Repeated measurements can otherwise let a model recognize the individual instead of generalizing to new individuals. [Chaibub Neto et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789029/).

Audit actual GUID overlap with the checkpoint's pretraining and checkpoint-selection populations too. The intended pipeline reserves healthy-BG leftovers for pretraining, but intended folder roles do not prove the actual run was disjoint. Pretraining exposure of test GUIDs must be disclosed; select an unexposed held-out subset if available. If previous model development used this test population, describe this pilot as exploratory reuse rather than a pristine final test. Check the provenance of the normalization statistics as well.

Use all four healthy subgroups versus both acidosis and HIE subgroups for the main binary question. Report results restricted to healthy-BG controls as a prespecified sensitivity analysis: healthy-no-BG controls differ in ascertainment, and unhealthy groups have `bg_label=True`. Report CS/no-CS strata where counts permit. These are descriptive checks, not additional fitted models or causal adjustment for delivery decisions.

For every split report unique recordings, segments, eligible anchors, class/subgroup counts, missingness and recording end-time distributions. Counts after late-window eligibility matter more than total shard size. If validation/test lacks either binary class, the specified experiment cannot estimate discrimination on that split; do not select another fold because its pictures look better.

### 4.2 Anchor timestamps, not segment-start labels

The loader trims signals but leaves `epoch` as the **untrimmed segment start** in seconds relative to delivery. For trimmed coefficient anchor index $a$, with trim $m$ minutes:

$$
t_{isa}=\mathrm{epoch}_{is}+60m+4a,\qquad
r_{isa}=-t_{isa}/3600.
$$

Here $r$ is hours before delivery. With $\mathrm{epoch}=-3600$, $m=1$, $a=150$, the anchor is at $-2940$ seconds, or **49 minutes before delivery**, not 60 minutes. Derive the four-second step from the stored decimation contract; confirm indexing with a known raw/feature example before exporting results.

Keep anchors with $0 < r \le 3$. Retain complete source segments and their history, even when they start earlier than three hours. A coarse loader filter can start at $-10800 - 1320$ seconds for the current 22-minute stored segments, followed by exact anchor filtering. Setting only `epoch_min=-10800` loses boundary-crossing segments. Do not crop the input to a three-hour or one-hour boundary and restart the filters.

Use the checkpoint's warm-up, source availability, `anchor_valid`, coverage and scored-anchor masks, with deterministic dense anchors (`anchor_phase=0`, `anchor_stride=1`) for collection. Conservatively reuse forecast-contributing support for this pilot. Thus the latent analysis also inherits forecast-availability exclusions; it is not every theoretically inferable pre-delivery state. Record that limitation.

For preservation forecast checks, verify every scored coefficient endpoint remains pre-delivery, including per-channel `target_forecast_shift` on legacy clocks. $\mathrm{epoch} < 0$ alone does not prove an entire segment or forecast is pre-delivery. Do not shift anchor timestamps to approximate filter-content times. **The stored UP timeline is canonical; apply no downstream timing correction.** Slow feature support and any checkpoint input alignment limit interpretation of rapid latent changes.

Deduplicate `(guid, epoch, anchor)` and inspect duplicate absolute anchor times within a GUID. If overlapping segments provide the same time, retain one deterministically using greater valid history, then a fixed segment-order tie-break. Never count them as independent observations. Warm-up/stride can leave real gaps between segment readouts; do not draw an apparently continuous trajectory over them.

### 4.3 Late bags and earlier trajectories

Define the supervised bag using $0 < r \le 1$ hour. Proposed eligibility, fixed before outcome comparisons: at least two contributing segments in that hour and a valid anchor within the final 30 minutes. Apply exactly the same rule to both classes; report exclusions. These are pragmatic coverage rules, not clinical onset criteria. Record the actual last observed time; never move the last available hour to the delivery landmark or extrapolate a missing terminal trace.

For segment $s$, mean-pool valid late anchors:

$$
e_{is}=\frac{1}{|A^{\rm late}_{is}|}\sum_{a\in A^{\rm late}_{is}}\mu^q_{isa}.
$$

Pool these segment vectors into one recording vector, mildly favoring more recent segments:

$$
\omega_{is}=2^{-\widetilde r_{is}/0.5},\qquad
v_i=\frac{\sum_s\omega_{is}e_{is}}{\sum_s\omega_{is}},
$$

where $\widetilde r_{is}$ is the median time-before-delivery of that segment's included anchors. This is a 30-minute half-life **within the final hour**, identical for healthy and adverse outcomes. It is a weighting preference, not a severity target. Segment averaging prevents dense anchor counts from dominating; each recording supplies one supervised loss. Do not concatenate time-bin vectors as classifier inputs in the first experiment.

Analyze the last three hours in six fixed bins: $(2.5,3]$, $(2,2.5]$, $(1.5,2]$, $(1,1.5]$, $(0.5,1]$, $(0,0.5]$ hours before delivery. Within each bin, average anchors within segments, then segments within recordings. A recording without late eligibility can appear in a separately labeled coverage/trajectory appendix, but not silently enter the paired primary comparison.

## 5. Smallest useful change to the model

### 5.1 Freeze everything except the posterior-mean output

Initialize from the pretrained checkpoint. Train only:

1. `model.posterior_head.delta_mu_head` (all its per-head linear modules when head-structured).
2. A new linear binary classifier `Linear(d_z, 1)` applied to the pooled posterior mean.

Freeze encoders, adapters, prior head, attention, posterior fusion/norms, variance heads, and decoder. Under the current 64-dimensional/four-head geometry, the mean-output modules contain **2,112 parameters** ($4\times(32\times16+16)$), plus 65 classifier parameters; derive and log the actual count for the loaded checkpoint.

Keep the backbone in `eval()` mode during this head-only fit to disable dropout, while enabling gradients on the mean-output parameters. `eval()` does not disable autograd. Do not put the student's trainable head under `no_grad()`. Optimizer parameters must be an explicit allowlist, not all model parameters.

This updates actual decoder-consumed `mu_post` coordinates. Merely fitting a classifier or a separate projection MLP while freezing all of `mu_post` would not satisfy the request to change the model's latent representation. Conversely, this narrow update does **not** adapt the upstream Transformer or attention patterns; a negative result is a limit of this small adaptation, not proof that the full architecture cannot learn the distinction.

With the chosen freeze set, `mu_prior`, both latent log-variances and attention weights should be identical before/after in deterministic evaluation. `mu_post`, the source-conditioned KL and full-branch forecasts may change. A changed KL does not demonstrate changed physiological coupling.

### 5.2 Objective: outcome discrimination plus local preservation

Compute per-coordinate mean $m_d$ and standard deviation $s_d$ from pretrained **training** latent anchors, giving recordings equal weight. Freeze these statistics for every comparison. Floor scales at $\max(10^{-3},\,0.1\,\operatorname{median}\{s_d:s_d>0\})$; if no dimension varies, report collapse rather than running the fit. This is separate from, and does not replace, the checkpoint's input normalization.

Let $S(x)=(x-m)/s$, and $v_i(\theta)$ be the pooled latent vector above. With one linear logit:

$$
\ell_i=w^\top S(v_i(\theta))+b,\qquad
\mathcal L_{\rm cls}=\frac12\sum_{c\in\{0,1\}}
\frac{1}{N_c}\sum_{i:y_i=c}\operatorname{BCEWithLogits}(\ell_i,y_i).
$$

Implement this as balanced sampling of **recordings**, or equivalent recording-level class weights, not both. Sample uniformly within each binary class; retain the natural acidosis/HIE mixture and disclose it. Keep natural prevalence in validation/test. Class-balanced training logits are association scores, not calibrated clinical risk estimates.

Use the pretrained model as a fixed teacher across valid anchors throughout the last three hours:

$$
\mathcal L_{\rm keep}=\operatorname{mean}_{i}\operatorname{mean}_{s\in i}
\operatorname{mean}_{a\in A^{3h}_{is}}
\frac1{d_z}\left\|\frac{\mu^q_{\theta,isa}-\mu^q_{0,isa}}{s}\right\|_2^2,
\qquad
\mathcal L=\mathcal L_{\rm cls}+0.1\mathcal L_{\rm keep}.
$$

The teacher has no gradients. The preservation term discourages gratuitous movement, including in earlier portions lacking temporal labels. It is an engineering regularizer, not a physiological smoothness assumption or a guarantee of preserved forecasting. Earlier latents may still change because the same parameters serve every time point.

Why a linear classification loss first? Its gradient with respect to a standardized bag vector is $(\operatorname{sigmoid}(\ell)-y)w$: it directly encourages discrimination along a latent direction, while the preservation term limits movement elsewhere. It does not guarantee compact Euclidean clusters. This distinction determines what the figures may establish.

Keep the original forecast loss out of the **first** fine-tuning objective to avoid decoder optimization, stochastic-loss scaling and pretraining schedule changes. Use forecast preservation as a validation gate instead (§6). This is supervised adaptation of a pretrained latent mean, not continuation of the original variational objective. In particular, do not accidentally restart its long KL warm-up or 2,000-step learning-rate warm-up through the existing trainer defaults.

### 5.3 Why not start with supervised contrastive learning?

SupCon explicitly brings same-class embeddings together and separates different-class embeddings. It is a reasonable next experiment if global geometric compactness is essential. [Khosla et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

Here its pair labels would be unreliable across early disease evolution, and pooled acidosis/HIE cases need not form one compact state cluster. It adds pair construction, temperature, loss weighting, and batch-size choices. Defer it. If later tried, operate on the same late **recording** bags, require same-class positives from different GUIDs, and apply the metric loss directly to normalized real latent summaries. Keep its projection separate from claims about raw `mu_post` if an extra projection head is used. Do not make all early HIE anchors positive pairs or negatives against healthy anchors.

## 6. Concrete run and selection protocol

| Item | Pilot default |
|---|---|
| Fold / seed | `fold_1` / `42`, declared before extraction |
| Supervised / preservation windows | Final 1 hour / final 3 hours |
| Baseline | Same bag representation and linear classifier, all model weights frozen |
| Fine-tuning | Mean-output layers + linear classifier only |
| Optimizer | AdamW; mean-output LR `1e-4`, classifier LR `1e-3`, weight decay `1e-4` |
| Batch | 8 distinct recordings, 4 per binary class; all available late segments per bag |
| Duration | Up to 10 epochs; stop after 3 epochs without improved eligible validation AUROC |
| Loss weights | Classification `1`, preservation `0.1`; no sweep in the first run |
| Numeric policy | Float32, gradient norm clip `1.0`, fixed stochastic seeds |
| Selection | Highest recording-level validation AUROC among checkpoints passing preservation checks; tie: lower validation BCE, then earlier epoch |

Fit the frozen baseline classifier first, with the same training bags and validation rule. Initialize fine-tuning's classifier from that baseline and the mean-output layers from the original checkpoint. Keep the original/frozen candidate as epoch zero: if adaptation adds no validation value, retain and report it rather than selecting an arbitrary changed model. Baseline head optimization can run cheaply on cached vectors until validation stalls; log its budget.

For a small class with fewer than four train recordings, reduce the batch and use accumulation; never manufacture distinct patients from repeated segments. If memory is tight, accumulate bag/recording losses or cache the frozen inputs to `delta_mu_head`. Such caching is valid only with upstream parameters fixed and dropout disabled. Verify cached and full-forward means agree before using it, and insert the learned weights into the actual model for final evaluation. Caching is an optional speed optimization, not a different representation.

**Preservation gate:** on the same prespecified validation GUIDs and anchor support, measure full-branch forecast MSE in the checkpoint's normalized coefficient space, averaging anchors/segments/recordings consistently. An initial engineering tolerance is at most a 10% increase over baseline MSE; declare it before training. Include healthy-BG and adverse-outcome examples and check healthy-only drift too. Also require finite means/losses and no substantial new posterior-delta saturation (proposed tolerance: no more than +5 percentage points on valid support). Record the tolerances as heuristic, not clinically validated.

For baseline and selected model, also report matched-policy Monte Carlo full-branch NLL and the source-conditioned KL diagnostics using the existing evaluation routines. Reuse identical random draws for before/after comparisons; start with eight draws, and check a larger draw count if a conclusion depends on a small NLL change. Training-path `pred_gap` uses asymmetric decoding policies and is not the primary preservation statistic. A fixed decoder alone does not preserve forecasts when its latent input moves.

Test remains unused until the baseline, fine-tuned checkpoint, scaling, plots and metrics are fixed. A failed validation pilot is still a result. Do not repeatedly open test plots while modifying hyperparameters.

## 7. Analysis that can distinguish real improvement from a pretty plot

### 7.1 Primary table: one observation per held-out recording

Compare frozen and adapted models on the **same eligible test GUIDs** and the same final-hour bags:

- AUROC and average precision (state adverse-outcome prevalence; the chance AP depends on it).
- Balanced accuracy at a threshold selected on validation, never test.
- Paired changes in those metrics, with 95% intervals from 1,000 bootstrap resamples of GUIDs, stratified by binary outcome. Use the same resampled GUIDs for both models. If patients contain multiple GUIDs, resample patients instead.
- Full-branch forecast MSE/NLL, standardized latent movement, latent covariance/effective rank and saturation before/after.

Intervals describe uncertainty conditional on this fold and fitting seed, not variation across training runs. Tiny HIE strata should show counts and broad intervals or descriptive points, not unsupported significance claims.

Use each model's fitted linear head for its main result; both have the same capacity. As a cheap geometry check independent of that head, classify using the nearest training-class centroid in the same fixed standardized $d_z$ space. Increased linear discrimination without improved centroid separation is possible and should be stated. Compute geometry metrics in full latent space, not on PCA/t-SNE coordinates.

Keep acidosis and HIE as separate plot colors and report their respective contrasts with the same healthy controls descriptively, even though training is binary. Do not select the better subgroup as the headline afterward.

### 7.2 Three main figures

**Figure 1 — actual latent space before and after.** Fit one two-component PCA on the concatenation of pretrained and adapted **training** recording-bin vectors, after the fixed training scaling. Weight/subsample so recordings, occupied bins and model versions have controlled contribution. Do not supply class labels to PCA. Transform validation/test with this single saved map; use identical axes for before/after panels. Report explained variance. Show final-hour recording bags colored healthy/acidosis/HIE, with a deterministic subset of paired before/after arrows. Reuse this map for trajectories. A pretrained-only PCA can be an optional sensitivity plot if movement in new directions is hidden.

**Figure 2 — separation along the learned latent direction.** Show test distributions of $w^\top S(v)+b$ before and after, explicitly labeled as a supervised axis. Include recording-level AUROC/AP with counts. A useful separation may be visible here when the leading PCs reflect other large sources of variation. The score is not an independently discovered biomarker and its sigmoid is not calibrated severity.

**Figure 3 — evolution over the last three hours.** For each model, apply its frozen final-hour classifier to each available recording-bin latent summary. Plot group means and GUID-bootstrap bands, with counts below each bin and a marked final-hour training region. Bin scores outside that region are exploratory applications outside the supervised window. Add a small, prespecified random sample of individual trajectories, with raw FHR/UP excerpts when available. Use signed hours $-3$ to $0$ on the display so delivery is at the right. Mark actual observations and gaps; never extend a line to delivery without data.

Do not independently refit PCA for every model/time bin and interpret axis changes as latent motion. Skip t-SNE/UMAP in the minimum deliverable. If a nonlinear embedding is added, make it an unsupervised, seed-fixed appendix with settings disclosed, and make no claim from relative distances between separate fits. t-SNE is a visualization of high-dimensional neighborhood structure, not a held-out discrimination test. [van der Maaten and Hinton, JMLR 2008](https://jmlr.org/papers/v9/vandermaaten08a.html).

### 7.3 Temporal question and simple controls

For the selected fine-tuned classifier, compute each recording's mean score in an early window $(2,3]$ hours and a late window $(0,1]$ hours, using the aggregation order above. Among recordings observed in both, calculate $\Delta_i = \mathrm{late}_i - \mathrm{early}_i$. Compare the distribution of $\Delta_i$ between adverse and healthy groups with a GUID bootstrap, and show acidosis/HIE separately. Also show the same calculation for the pretrained baseline. This within-recording analysis reduces composition changes caused by different patients contributing to different bins; it does not remove coverage selection or confounding.

An increase toward delivery would be **consistent with** changing outcome-associated signal, not proof of physiological worsening. No increase is also plausible. A clinical onset or severity validation would require independently annotated time-resolved information, which is outside this pilot.

Minimum controls beyond the frozen baseline:

1. **Shuffled training labels:** rerun the tiny adaptation once with a fixed permutation at GUID level, preserving every segment's assigned recording label. Select using validation labels permuted independently within validation; evaluate once against true held-out test labels. Expect no consistent held-out gain. One shuffled run is a leakage/overfit sanity check, not a permutation p-value; a formal p-value would require many full refits including selection.
2. **Time/coverage and subgroup checks:** compare time distributions, valid coverage, end-of-recording times and CS/BG strata by class; report the healthy-BG restriction and CS strata when supported. Color the same PCA by these variables. A cloud separated mainly by ascertainment or signal quality limits the outcome interpretation.
3. **FHR-only readout:** report a cheap frozen linear probe on pooled `mu_prior` using the same split and protocol if claiming any benefit from the combined branch. `mu_prior` must remain unchanged in the proposed adaptation. Superior `mu_post` discrimination alone does not establish UP-specific information; a stronger source-use claim would additionally need source-null/shuffle controls and matched forecasts.

Optional second-seed repetition is more useful than trying many projection settings, but is not required to finish this first pilot.

## 8. Minimal implementation plan in this repository

The existing [evaluation entry point](eval/run.py) and [binding](eval/binding.py) already rebuild the Transformer CFS checkpoint through the shared CFS evaluator. Reuse strict loading, input assembly, and mask computation.

Important gaps in the existing evaluator:

- [The `latent` analysis](../lag_attn_cfs/eval/analyses/latent.py) reports KL spectra and variance/saturation diagnostics. It is not a healthy/adverse latent-coordinate embedding analysis.
- [Collection](../lag_attn_cfs/eval/collect.py) exports scalar/vector readouts, but does not currently provide a complete per-anchor `mu_post`/`mu_prior` coordinate table. Its `per_anchor_vectors.npz` filename is not proof that these latents are saved.
- [Cohort time bins](../lag_attn_cfs/eval/cohort.py) use segment-start `epoch`. This pilot needs the trim- and anchor-aware timestamp above. Add a dedicated column/path instead of silently changing the meaning of existing reports.
- The current training load list omits clinical `target`; explicitly load outcome/quality metadata or join an audited GUID manifest. The original pretraining driver is not already a bag-level classification trainer.

All new implementation belongs inside `teb_vae/lag_attn_transformer_cfs/latent_pilot/`, including tests, configuration and generated run artifacts. This document stays at its current path as the single design and task-tracking entry point. The detailed layout and ordered implementation checklist are in §11. The following responsibilities are implemented inside that folder (these files are **proposed**, not created by this document):

| File | Responsibility |
|---|---|
| `latent_pilot/train.py` | Recording bags; trainable-parameter allowlist; baseline and tiny adaptation; validation selection; standalone pilot checkpoint |
| `latent_pilot/analyze.py` | Exact latent comparisons, fixed projections, recording bootstrap, trajectories and tables |
| `latent_pilot/configs/pilot.yaml` | Explicit checkpoint/fold paths, outcome map, windows, eligibility, optimizer, seed, gates and output path |

Use a separate pilot namespace for new configuration keys: the existing model-constructor signature filtering can silently ignore unrelated YAML leaves. Preserve original checkpoint stamps and model kwargs. Store the classifier/scaler and pilot metadata separately or in a clearly defined wrapper; do not send extra classifier keys through strict loading of the base model. Save a base-model-compatible state dict with the adapted mean-head weights for reuse in the forecast evaluator. Never overwrite the pretrained checkpoint.

Suggested output directory contents:

```text
latent_pilot/runs/fold_1/seed_42/<run_id>/
  protocol.yaml                 # choices, original checkpoint hash/config, software revision
  split_manifest.csv            # GUID, split, outcome/subgroup, provenance and eligibility
  coverage.csv                  # counts and actual temporal coverage, including exclusions
  pretrained_latents.npz         # keyed means, log-variances and timestamps on exact support
  finetuned_latents.npz
  latent_index.parquet           # GUID/segment/anchor keys aligned with array rows
  pilot_checkpoint.pt            # adapted weights, head, scale and fit metadata
  metrics.csv                    # recording-level baseline/after/control comparisons
  per_recording_bin.parquet
  figures/                      # fixed PCA, supervised-axis distributions, trajectories
  report.md                      # measured results, exclusions and limits
```

A single-GPU custom loop is sufficient. Implement the validation, smoke and extraction/fit/evaluation stages so the user can run them in order after all code is written. Do not execute those stages during implementation. Do not adapt the raw preprocessing, retrain scattering features, change forecast clocks, or add an all-fold experiment to this preliminary work.

## 9. Tests before trusting results

These are focused checks to implement as test code. Only the small synthetic logic subset defined in §11.1 may be run by the coding agent; checkpoint/model/gradient/integration and experiment checks belong to the user on the other machine. No new executable tests are needed merely to edit this proposal.

| Check | What should be established |
|---|---|
| Provenance and split | No prohibited GUID/patient overlap; label consistency; input statistics have documented fitting population |
| Timestamp example | The $\mathrm{epoch}=-3600$, trim-one-minute, anchor-150 example gives 49 minutes; boundary-crossing segments retained; post-delivery endpoints excluded |
| Mask and identity | No warm-up/padded/invalid anchors; no unintended duplicate times; before/after use exactly the same exported keys |
| Bag reduction | Synthetic unequal segment lengths give the intended segment-then-recording weighting; duplicating anchors does not multiply a recording's supervised weight |
| Gradient reach | An update changes permitted mean-output weights and actual `mu_post`; frozen weights, `mu_prior`, log-variances and attention remain equal |
| Determinism and reload | Repeated `eval()` mean extraction agrees; optional cache equals full forward; saved/reloaded pilot reproduces means and logits |
| Fit isolation | Scaler/PCA/class centroids use training data only; early stopping/threshold use validation only; test labels never enter training or projection fitting |
| Preservation | Forecast error and posterior-delta saturation satisfy the declared validation gates; no global latent collapse |

## 10. How to write the conclusion

A positive preliminary result would show a paired held-out improvement over the frozen baseline, visible separation in a disclosed latent view or learned direction, acceptable forecast/latent preservation, and no obvious explanation from time coverage or clinical subgroups. Stronger geometric evidence would include improved nearest-centroid performance in the original standardized latent space. A confidence interval spanning no improvement should be described as inconclusive, even if the picture is attractive.

Use wording such as:

> On one prespecified fold, a small supervised adaptation of the posterior-mean layers improved [measured metric] for late-recording healthy versus adverse-outcome summaries. Earlier trajectories showed [observed pattern]. These are exploratory outcome-associated latent changes; they do not assign physiological state or injury onset to individual time points.

If only training separation improves, report overfitting. If frozen and adapted models perform similarly, report that pretraining already captured the available signal or that this restricted adaptation added no demonstrated benefit. If discrimination improves but forecasts or latent diversity degrade, report the tradeoff rather than calling the representation uniformly better. Decide whether to unfreeze one final Transformer block or try late-bag contrastive learning only after completing and documenting this minimal comparison.

## 11. Coding-agent implementation checklist and progress tracker

### 11.1 Start here: write everything first; execution belongs to the user

**Read this entire document before implementing.** The scientific protocol remains §§1–10. Complete the code, configuration, test definitions and documentation first; then perform only the minimal synthetic logic check in LP-16. Implement the complete pipeline, including analysis, figure and report generators, and hand it to the user without running the experiment. **Agents do not have access to the execution machine and must not depend on gaining access.**

The user requires all new implementation inside `teb_vae/lag_attn_transformer_cfs/latent_pilot/`. This existing Markdown document is the intentional exception: it stays here as the design and live tracker.

Working rules:

1. Read applicable instructions, source files and existing configuration; preserve unrelated edits. Write and review code locally. Do not launch the full runner, CLI help, model/checkpoint loading, dataset preflight, smoke/integration tests, extraction, training, evaluation or report generation. Do not request remote access or assume production files are locally available. Document dependency setup for the user's machine instead of installing a production environment here.
2. Write focused tests alongside the behavior they check. After coding, run only the small `tests/logic/` subset described below, using synthetic in-memory values and existing lightweight dependencies. Leave all model/data/integration tests for the user. Record exactly which tests ran; do not describe the entire suite as passing.
3. Import existing model/data/evaluation code through small local adapters. Keep all new source, configs, test definitions, fixture-generator code and eventual artifacts in `latent_pilot/`; keep the original checkpoint/datasets in place and reference them by path. Do not change original training defaults or copy whole sibling modules.
4. Complete every required coding task before handing off execution. Missing real paths do not prevent coding: provide explicit placeholders and implement clear runtime validation. Do not manufacture clinical results or silently use smoke data as production data.
5. Record source-file evidence and what was reviewed for each completed task. Keep implementation, minimal logic-test status and target-machine execution status separate. Completion means **implemented and reviewed, minimal logic results disclosed, target-machine execution pending**.
6. Preserve the fixed one-fold protocol and scope. Any necessary design deviation needs a written reason and its effect on interpretation. Do not add sweeps, larger unfreezing, onset labels or new losses.

**Minimal agent-side logic check:** keep one small, fast suite covering timestamp/window boundaries, label/split consistency on artificial IDs, bag-reduction weighting, training-only scaler statistics on small arrays, and dictionary/config precedence without opening production files. Pure selection/tie/fallback logic may be included if similarly small. Use hand-checkable values; no model construction, tensor optimization, checkpoint I/O, clinical HDF5, GPU, subprocess runner launch, plots or generated dataset fixtures. Keep this subset import-isolated: parent `conftest.py` and imported helper modules must not initialize the model or pull in the production pipeline. A temporary directory is acceptable for a tiny config-only fixture if necessary. Run the focused subset once after coding and rerun only affected checks after fixing a failure; do not broaden it into an end-to-end validation campaign.

**Current handoff state:** LP-01 through LP-16 written; the coding checklist is complete. `RUN_ARGS`, `main()`, the settings resolver and all nine stage handlers are written, so every stage is selectable from the dictionary and from the command line. Nothing in the coding tracker remains open. **pytest is not installed on the coding machine**, so the logic tests were exercised by importing each module and calling every `test_*` function directly with a throwaway stand-in for `pytest.raises`/`approx`/`mark.parametrize` (289 checks, 0 failed); the user runs the real command, `python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/logic -q`, on the execution machine. The tests that build the real net are deliberately **not** in that subset and live in `latent_pilot/tests/test_model_contract.py`, `latent_pilot/tests/test_extraction.py`, `latent_pilot/tests/test_preservation.py`, `latent_pilot/tests/test_baseline_fit.py`, `latent_pilot/tests/test_adaptation_fit.py`, `latent_pilot/tests/test_persistence.py`, `latent_pilot/tests/test_controls.py`, `latent_pilot/tests/test_figures.py` and `latent_pilot/tests/test_runner.py`, which need no clinical data, no checkpoint file and no GPU, but do need pytest and the repository environment. `latent_pilot/tests/test_fixtures.py` and `latent_pilot/tests/test_smoke.py` additionally need the generated fixtures and skip with the command that writes them: `python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate`. The minimal synthetic subset is completed in LP-14 and run once more in LP-16. Figure rendering is not in the logic subset either -- the subset draws nothing -- and is exercised by `latent_pilot/tests/test_figures.py` and again by the smoke scenario, neither of which has been run here. The actual experiment remains `USER_TO_RUN_ON_OTHER_MACHINE`.

### 11.2 Single-folder layout and IDE runner contract

Use this small package layout; adjacent responsibilities may share a module if that simplifies the implementation.

```text
teb_vae/lag_attn_transformer_cfs/
  LATENT_CLASS_FINETUNING_PILOT.md       # design + coding tracker + user run guide
  latent_pilot/
    __init__.py
    .gitignore                         # future runs/caches, not source configs
    run.py                             # RUN_ARGS, main(), CLI, guarded __main__
    config.py                          # strict settings and path/config resolution
    data.py                            # manifest, labels, times, masks, recording bags
    model.py                           # strict loading and trainable-head adapter
    extract.py                         # keyed extraction and training-only scaling
    train.py                           # baseline, tiny adaptation, shuffled control
    evaluate.py                        # preservation, held-out metrics, bootstrap
    analyze.py                         # temporal summaries, full-space geometry, PCA
    report.py                          # figure/report generation functions
    configs/
      pilot.yaml                       # production template with explicit placeholders
      smoke.yaml                       # small, explicitly nonclinical configuration
    tests/
      logic/                           # minimal synthetic agent-side checks after coding
      ...                              # model/integration test code: user runs remotely
      fixtures/                        # small fixture definitions/generator code
    runs/fold_1/seed_42/<run_id>/
      ...                              # created only when the user runs the pipeline
```

Use the repository environment/dependencies where possible. Document any necessary setup locally. Do not create a root dependency file or external experiment-logging integration for this pilot.

**IDE Run button is a first-class requirement.** The existing [trainer.py](trainer.py) uses `RUN_CONFIG`, a guarded `__main__`, repository-root import bootstrapping, and path resolution independent of the IDE working directory. Follow that launch pattern, extending it to an editable **dictionary** rather than requiring CLI arguments.

Provide an interface like this in `latent_pilot/run.py` (design sketch, not existing executable code):

```python
RUN_ARGS = {
    "config_path": "teb_vae/lag_attn_transformer_cfs/latent_pilot/configs/pilot.yaml",
    "stage": "all",
    "device": None,       # None uses the config; e.g. "cuda:0" or "cpu"
    "run_dir": None,      # None creates a run under latent_pilot/runs/
    "resume": False,
    "overrides": {
        # Optional pilot-config overrides; edit these or the YAML before running.
        # "checkpoint_path": "/absolute/path/to/pretrained.ckpt",
        # "statistics_path": "/absolute/path/to/matching_stats.hdf5",
        # "train_shards": ["/absolute/path/to/fold_1/train/example.hdf5"],
        # "val_shards": ["/absolute/path/to/fold_1/val/example.hdf5"],
        # "test_shards": ["/absolute/path/to/fold_1/test/example.hdf5"],
    },
}


def main(*, config_path, stage="all", device=None, run_dir=None,
         resume=False, overrides=None):
    """Resolve settings and dispatch the requested stage; never parse sys.argv here."""
    ...


if __name__ == "__main__":
    # No CLI arguments: use RUN_ARGS as edited above.
    # Explicit CLI arguments override the corresponding dictionary entries.
    main(**resolve_run_args(RUN_ARGS, argv=sys.argv[1:]))
```

Implement the helper and imports; do not leave an ellipsis in the final runner. Keep `RUN_ARGS` near the bottom of the file with clear per-key comments. A user must be able to edit it and hit Run without configuring an IDE working directory, writing another launcher, or supplying CLI flags.

Launch/config semantics:

- Support direct execution of `run.py`, module execution, and programmatic `main(**RUN_ARGS)`. Direct-file execution needs a guarded repository-root `sys.path` bootstrap **before** package imports. The pilot runner is one directory deeper than `trainer.py`; derive its root correctly.
- Importing any pilot module must not launch work, read production datasets, create run directories, spawn processes or parse command-line arguments.
- Resolve repository-relative paths against the repository root; preserve absolute paths. Apply this consistently to config, dataset, checkpoint and output paths, independent of the working directory. Runtime output paths must remain in the pilot folder.
- Keep checked-in defaults portable: do not bake in this coding machine's home directory, virtual environment, GPU IDs or mounted data paths. The user will copy/check out the repository and set machine-specific values in YAML or `RUN_ARGS` on the execution machine. Write dependency/setup notes and runtime error messages sufficient for that transfer; no remote service or agent connection should be needed.
- Use one settings resolver for every launch mode. Resolve built-in pilot defaults, then YAML, then dictionary overrides, then explicitly supplied CLI overrides; non-`None` top-level dictionary values such as `device` override YAML, and explicit CLI values override those. Treat parser defaults as unspecified, not as overrides. Specify merge behavior, including replacing lists rather than concatenating them.
- Offer CLI equivalents for config/stage/device/run directory/resume and an optional structured config override mechanism. `--config` chooses the YAML before applying the same explicit dictionary/CLI overrides. Do not silently change checkpoint architecture/preprocessing through pilot overrides.
- Neither `main()` nor the config resolver may mutate the global `RUN_ARGS` dictionary or its nested objects. Persist the effective arguments and resolved protocol when the user starts a run.

Supported stages: `tests`, `smoke`, `preflight`, `extract`, `baseline`, `finetune`, `control`, `evaluate`, `report`, `all`. **Every stage must be selectable through `RUN_ARGS["stage"]` and the same `__main__`.** This includes tests and the smoke check, so CLI/pytest commands are optional conveniences.

`all` must implement this future execution order:

```text
tests -> smoke -> preflight -> extract(train/validation)
      -> baseline -> finetune -> control
      -> freeze selection/settings -> evaluate(test + analyses) -> report
```

`tests` can invoke pytest using the current Python interpreter and a fixed repository-root working directory. `smoke` uses its dedicated fixture config/output and an internal pipeline dispatcher; it must not recursively invoke `all` or overwrite production arguments/artifacts. Both stop the sequence on failure. Individual runtime stages consume saved dependencies, reject incompatible artifacts and fail clearly if preceding stages have not completed. Do not silently train a missing model when the user requested `report`.

Test data can supply read-only label/coverage metadata during runtime preflight, but never fitting inputs or selection metrics. `extract` collects train/validation only; `evaluate` collects test latents after models, thresholds, controls and analysis settings are fixed. Fit PCA/class centroids on saved training data before applying them to test. The agent writes this ordering; only the user executes it.

### 11.3 Runtime inputs: write placeholders and validation

Read available configuration/source metadata to understand these inputs. Record known paths when already supplied; otherwise leave explicit placeholders for the user. Do not load a checkpoint, scan clinical datasets or launch preflight merely to fill this table.

| Input | Current value | Code must support |
|---|---|---|
| Pretrained checkpoint path and identity | User to supply | Strict reconstruction and runtime hash/provenance record |
| Matching saved config/model kwargs | User to supply or derive from run metadata at runtime | Architecture, feature/clock/trim contract |
| Fold-1 train/validation/test shard lists | User to supply | Fixed splits, subgroup and GUID manifest |
| Matching input statistics and provenance | User to supply | Correct preprocessing and exposure disclosure |
| Pretraining/checkpoint-selection GUID provenance | User to supply if available | Known/unknown exposure tracking |
| Patient/delivery grouping beyond GUID | Optional; disclose GUID-only grouping if absent | Split validation and clustered resampling |
| Python environment/device | Configurable | Tests/smoke and single-device real run |

At runtime, unknown provenance stays unknown, not an assertion of no overlap. Known leakage must prevent an unsupported clean-holdout claim. Missing essential files must raise a precise error naming the setting. These are behaviors to implement, not prerequisites to finish writing code.

### 11.4 Coding-only master tracker

Statuses: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`; optional extensions may be `DEFERRED`. For LP-01–LP-15, `DONE` means code/documentation written and reviewed by inspection. LP-16 also records the minimal logic check, including any unavailable dependency. Full integration/experiment status remains `USER_TO_RUN_ON_OTHER_MACHINE` regardless of local logic results. Use checkboxes for partial writing progress and record separate source/test evidence in §11.7.

| ID | Coding task | Depends on | Implementation status | Source evidence / remaining work |
|---|---|---|---|---|
| LP-01 | Write package skeleton and integration notes | — | DONE | `latent_pilot/{__init__,config,data,model,extract,train,evaluate,analyze,report,run}.py`, `latent_pilot/.gitignore`, `latent_pilot/tests/{,logic/,fixtures/}__init__.py`, `latent_pilot/configs/`. Modules are documented stubs; behaviour lands in LP-02–LP-15 |
| LP-02 | Write strict config and dictionary/CLI argument resolution | LP-01 | DONE | `latent_pilot/config.py` (`DEFAULTS`, `STAGES`, `STAGE_INPUTS`, `resolve_settings`, `require_inputs`, `stage_plan`, `run_id`, `run_directory`, `protocol_record`, `new_stage_state`, `PilotConfigError`), `latent_pilot/configs/{pilot,smoke}.yaml`, `latent_pilot/run.py` (`RUN_ARGS`, `RUN_ARG_DEFAULTS`, `build_parser`, `resolve_run_args`, `main`, guarded `__main__`). `STAGE_HANDLERS` is empty until LP-15 |
| LP-03 | Write cohort/label/provenance manifest code | LP-02 | DONE | `latent_pilot/data.py` (`CLINICAL_FIELDS`, `REQUIRED_LOAD_FIELDS`, `pilot_loader_config`, `segment_frame`, `binary_outcome`, `recording_frame`, `exclusion_counts`, `check_split_disjoint`, `load_guid_list`, `attach_patient_groups`, `exposure_record`, `require_both_classes`, `coverage_summary`, `write_manifest`, `write_coverage`), `latent_pilot/tests/logic/test_cohort.py` (19 checks). The temporal half of `data.py` is LP-04 |
| LP-04 | Write temporal support and recording-bag code | LP-03 | DONE | `latent_pilot/data.py` (`step_seconds`, `trim_seconds`, `anchor_seconds`, `hours_before_delivery`, `segment_span_seconds`, `coarse_epoch_min`, `in_window`, `contributing_support`, `max_forecast_shift`, `latest_scored_seconds`, `add_anchor_times`, `mark_window_and_delivery`, `deduplicate_anchors`, `retained`, `anchor_exclusion_counts`, `bin_edges`, `assign_time_bins`, `segment_means`, `recency_weights`, `recording_means`, `recording_bags`, `recording_bin_means`, `window_means`, `late_eligibility`), `latent_pilot/tests/logic/test_temporal.py` (31 checks) |
| LP-05 | Write strict model loading and gradient allowlist | LP-02 | DONE | `latent_pilot/model.py` (`LoadedCheckpoint`, `load_pilot_checkpoint`, `geometry_record`, `statistics_record`, `LatentClassifier`, `mean_head_module`, `mean_head_parameters`, `freeze_for_pilot`, `check_pilot_mode`, `parameter_groups`, `describe_trainable`, `frozen_teacher`, `forward_inputs`, `deterministic_outputs`, `compare_outputs`, `assert_invariants`, `INVARIANT_OUTPUTS`, `ADAPTED_OUTPUTS`), `latent_pilot/tests/test_model_contract.py` (22 checks, user-run) |
| LP-06 | Write latent export and fixed training-scaler code | LP-04, LP-05 | DONE | `latent_pilot/extract.py` (`LatentExtraction`, `extract_split`, `support_fingerprint`, `check_compatible`, `assert_same_keys`, `check_split_allowed`, `save_extraction`, `load_extraction`, `LatentScaler`, `LatentCollapse`, `fit_scaler`, `save_scaler`, `load_scaler`), `latent_pilot/tests/logic/test_scaler.py` (15 checks), `latent_pilot/tests/test_extraction.py` (11 checks, user-run). Frozen-fusion caching **omitted**, as the task permits |
| LP-07 | Write preservation metrics and candidate-gate code | LP-04, LP-05 | DONE | `latent_pilot/evaluate.py` (`SCORE_COLUMNS`, `GATE_STRATA`, `ZERO_BASELINE_RULE`, `GateEvaluationError`, `PreservationReading`, `outcome_map`, `gate_subset`, `save_gate_subset`, `load_gate_subset`, `preservation_pass`, `_support_digest`, `_preservation_record`, `save_preservation`, `load_preservation_record`, `GateResult`, `gate_decision`, `nll_convergence`), `latent_pilot/config.py` (`gates.subset_recordings`), `latent_pilot/configs/{pilot,smoke}.yaml`, `latent_pilot/tests/logic/test_gates.py` (25 checks), `latent_pilot/tests/test_preservation.py` (15 checks, user-run) |
| LP-08 | Write frozen linear baseline fitting code | LP-06 | DONE | `latent_pilot/train.py` (`BASELINE_NAME`, `FITTING_SPLITS`, `Bags`, `build_bags`, `balanced_bce`, `is_better`, `BaselineFit`, `fit_baseline`, `_labels_for`, `save_fit`, `load_fit`), `latent_pilot/evaluate.py` (`auroc`, `binary_cross_entropy`, `balanced_accuracy`, `select_threshold`), `latent_pilot/tests/logic/test_baseline.py` (24 checks), `latent_pilot/tests/test_baseline_fit.py` (16 checks, user-run) |
| LP-09 | Write tiny adaptation and checkpoint-selection code | LP-06–LP-08 | DONE | `latent_pilot/train.py` (`RecordingSource`, `SegmentSupport`, `RecordingPlan`, `build_plans`, `epoch_batches`, `recording_terms`, `AdaptationFit`, `fit_adaptation`, `_mean_head_state`, `_load_mean_head_state`, `_reference_gate`, `_verify_teacher`), `latent_pilot/model.py` (`to_device`), `latent_pilot/tests/logic/test_adaptation.py` (18 checks), `latent_pilot/tests/test_adaptation_fit.py` (12 checks, user-run) |
| LP-10 | Write artifact save/load/resume and selection locking | LP-08, LP-09 | DONE | `latent_pilot/config.py` (`SELECTION_LOCK_FILENAME`, `RERUNNABLE_STAGES`, `RunStateError`, `settings_differences`, `write_protocol`, `read_protocol`, `write_stage_state`, `read_stage_state`, `mark_completed`, `mark_failed`, `require_completed`, `open_run`, `lock_selection`, `read_selection_lock`, `require_selection_locked`), `latent_pilot/train.py` (`PilotCheckpoint`, `save_adapted`, `load_adapted`, `apply_adapted`, `export_base_checkpoint`, `_refuse_source_path`), `latent_pilot/tests/logic/test_run_state.py` (15 checks), `latent_pilot/tests/test_persistence.py` (12 checks, user-run) |
| LP-11 | Write held-out metrics, bootstrap and control code | LP-07, LP-10 | DONE | `latent_pilot/evaluate.py` (`METRIC_NAMES`, `CONFIDENCE`, `MIN_UNITS`, `SUBGROUPS`, `average_precision`, `recording_metrics`, `bootstrap_units`, `_strata`, `paired_bootstrap`, `subgroup_rows`, `subgroup_table`, `coverage_contrast`, `control_disclosure`), `latent_pilot/train.py` (`CONTROL_NAME`, `PRIOR_PROBE_NAME`, `permute_outcomes`, `control_recordings`, `fit_control_baseline`, `fit_prior_probe`), `latent_pilot/tests/logic/test_metrics.py` (26 checks), `latent_pilot/tests/test_controls.py` (8 checks, user-run) |
| LP-12 | Write temporal/geometry/PCA analysis code | LP-06, LP-10 | DONE | `latent_pilot/analyze.py` (`SCORE_COLUMN`, `PROJECTION_FILENAME`, `bin_summaries`, `score_frame`, `supervised_bins`, `group_bands`, `window_scores`, `_difference_ci`, `paired_contrast`, `class_centroids`, `nearest_centroid_scores`, `effective_rank`, `covariance_summary`, `movement_summary`, `Projection`, `projection_weights`, `fit_projection`, `save_projection`, `load_projection`), `latent_pilot/data.py` (`eligible_outcomes`, `eligible_anchors`), `latent_pilot/tests/logic/test_analysis.py` (29 checks). Everything here is hand-checkable without a model, so LP-12 needs no user-run test file |
| LP-13 | Write figure and report generators | LP-11, LP-12 | DONE | `latent_pilot/report.py` (`FIGURE_DIRNAME`, `REPORT_FILENAME`, `FIGURE_LATENT_SPACE`, `FIGURE_SUPERVISED_AXIS`, `FIGURE_TRAJECTORIES`, `PLOT_CLASSES`, `MISSING`, `CAPTIONS`, `RUNNER_MODULE`, `configure_figures`, `class_palette`, `variance_caption`, `axis_label`, `score_scale_note`, `select_traces`, `with_class_names`, `figure_latent_space`, `figure_supervised_axis`, `signed_bin_hours`, `figure_trajectories`, `reproduction_commands`, `build_report`, `write_report`), `latent_pilot/config.py` (`figure_format`), `latent_pilot/configs/pilot.yaml`, `latent_pilot/tests/logic/test_report.py` (39 checks). Figure rendering itself is exercised by the LP-14 smoke scenario, so LP-13 adds no user-run test file |
| LP-14 | Write focused tests and end-to-end fixture scenario | LP-02–LP-13 | DONE | `latent_pilot/tests/fixtures/generate.py` (`GENERATED_ROOT`, `SPLITS`, `SMOKE_SUBGROUPS`, `LATE_EPOCHS`, `EARLY_EPOCHS`, `split_guid`, `segment_epochs`, `write_split_shards`, `write_checkpoint`, `generate`, `manifest_matches`, guarded `__main__`), `latent_pilot/tests/conftest.py` (`smoke_settings`, `smoke_fixtures`, lazy imports so the logic subset stays isolated), `latent_pilot/tests/logic/test_settings.py` (42 checks), `latent_pilot/tests/logic/test_report.py` (+2 gap checks), `latent_pilot/tests/test_fixtures.py`, `latent_pilot/tests/test_runner.py`, `latent_pilot/tests/test_smoke.py`, `latent_pilot/tests/test_figures.py` (all user-run). The smoke scenario and the two stage-registry assertions state the dispatcher contract LP-15 completes |
| LP-15 | Wire all stages into RUN_ARGS/main/__main__ | LP-02–LP-14 | DONE | `latent_pilot/run.py` (`PIPELINE_STAGES`, `PRETRAINED`, `ADAPTED`, `GATE_REFERENCE`, the artifact filename constants, `_load_checkpoint`, `_loader_config`, `_loader`, `_extraction_name`, `_attach_eligibility`, `_rewrite_cohort`, `_bags`, `_bin_summaries`, `stage_tests`, `stage_smoke`, `stage_preflight`, `stage_extract`, `stage_baseline`, `stage_finetune`, `stage_control`, `stage_evaluate`, `stage_report`, `STAGE_HANDLERS`, `run_pipeline`, `main` on `open_run`/`mark_completed`/`mark_failed`), `latent_pilot/data.py` (`RECORDINGS_FILENAME`, `SEGMENTS_FILENAME`, `write_cohort_table`, `read_cohort_table`, `split_loader`). Reviewed by reading source and by static checks over the module: no module-level call, every stage registered, every module attribute referenced exists, and every call into the pilot modules binds against its real signature |
| LP-16 | Minimal logic check and other-machine execution handoff | LP-15 | DONE | §12 rewritten as the execution guide (requirements, the five paths to fill, Run-button first and CLI second, the artifact each stage writes, resume/re-report/refusal semantics, what to keep when a stage fails, and the handoff); `latent_pilot/configs/pilot.yaml` (where a run writes), `latent_pilot/configs/smoke.yaml` (the fixture-generator command), `latent_pilot/analyze.py` (docstring no longer calls an implemented module a skeleton), `latent_pilot/report.py` (`FIGURE_COVERAGE_SPACE`, `_scatter_panel`, `coverage_groupings`, `figure_coverage_space` -- the §7.3 coverage control's recoloured view, which was the one gap the review found), `latent_pilot/run.py` (renders it), `latent_pilot/tests/logic/test_report.py` and `latent_pilot/tests/test_figures.py` (its checks). Minimal logic subset: **289 checks, 0 failed**, via the pytest stand-in described in §11.1; everything else remains unexecuted |

There is no agent task to execute the experiment or produce measured results. Only minimal synthetic logic-test evidence is collected locally; full test results, model artifacts and clinical findings come from the user's §12 execution. All coding can finish with target-machine paths still unfilled.

### 11.5 Ordered coding tasks and static completion criteria

For LP-01–LP-15, **write** the stated functions, assertions and test definitions first. LP-16 permits the minimal isolated logic check only. Full runtime acceptance expectations remain §§4–9 and the integration tests for the user to execute.

#### LP-01 — Write the skeleton and integration notes

- [x] Create the local package skeleton and `.gitignore` from §11.2; do not create populated run/cache directories.
- [x] Read linked source APIs and record reusable symbols for checkpoint loading, CFS input construction, masks, labels, targets and forecast evaluation in local adapter docstrings or this log.
- [x] Document runtime inputs and their provenance handling without requiring actual data discovery. Derive dimensions/clocks from saved metadata in the planned interfaces rather than hard-coding current YAML examples.
- [x] Read existing environment/dependency declarations and document the intended environment/setup without installing packages or running imports.

**Code complete when:** the folder boundary, adapter responsibilities and configuration inputs are documented and the package skeleton is written.

#### LP-02 — Write config validation and dictionary/CLI resolution

- [x] Write strict pilot schema/defaults and production/smoke templates, including the §6 defaults, finite baseline budget, bootstrap/MC settings, gates, device and paths. Unknown keys and unsupported settings need explicit runtime errors.
- [x] Write `RUN_ARGS`, `main`'s signature and `resolve_run_args` according to §11.2, including precedence, deep-copy behavior and identical dictionary/CLI validation.
- [x] Write repository-root path normalization and output-containment checks; unresolved production placeholders should fail only when the user requests stages needing them.
- [x] Write run-ID, resolved-protocol and stage-state construction functions with no import-time side effects. Tests/smoke must not require production paths to resolve.

**Code complete when:** configs and resolver code cover no-argument IDE, explicit CLI and programmatic calls; source review confirms no executable work occurs on import. CLI help is not run by the agent.

#### LP-03 — Write cohort, labels and provenance handling

- [x] Write readers for GUID/segment identity, valid outcome codes, CS/BG flags and available onset metadata. Reject inconsistent supervised labels and count exclusions; never pass clinical metadata to model inputs.
- [x] Write fixed-fold split/group checks, duplicate detection and known/unknown pretraining/selection/statistics exposure handling.
- [x] Write manifest serialization for source shard/index, split, outcome/subgroup, eligibility and exclusion reason.
- [x] Write coverage/class-count/end-time summaries and runtime checks for missing binary classes; represent unavailable strata explicitly.

**Code complete when:** readers, validators and serializers exist, and fixture test definitions cover conflicting labels, invalid target zeros, split overlap and recording grouping.

#### LP-04 — Write anchor support and recording bags

- [x] Write whole-segment selection and exact $\mathrm{epoch}+\mathrm{trim\_seconds}+\mathrm{step\_seconds}\times\mathrm{anchor}$ timestamps, with a conservative boundary-crossing coarse filter.
- [x] Write an adapter to deterministic dense anchors and the existing warm-up/source/coverage/padding/contributing-support rules. Use one support policy for all model versions.
- [x] Write post-delivery forecast-endpoint checks for every scored channel, including checkpoint shifts, and serialize exclusion reasons.
- [x] Write deterministic key/absolute-time deduplication and late eligibility on retained support.
- [x] Write final-hour segment means and recency-weighted bags, three-hour teacher support, six half-hour bins and paired early/late windows.
- [x] Write known-answer tests for 49 minutes, $r\in\{0,0.5,1,3\}$ boundaries, crossing segments, shifted forecasts, missing bins and unequal segment lengths.

**Code complete when:** support and reduction functions implement §4, test cases encode the boundary answers, and source inspection finds no severity labels, time interpolation or monotonic constraint.

#### LP-05 — Write checkpoint reconstruction and gradient restrictions

- [x] Write strict `SeqVaeLagAttnTrfCfs` reconstruction from saved class/config/kwargs, matching statistics and resolved geometry.
- [x] Write the classifier wrapper and explicit optimizer allowlist for `posterior_head.delta_mu_head` and the classifier only.
- [x] Write evaluation-mode handling for frozen paths and detached teacher outputs, retaining gradients through student mean heads. Derive/log actual parameter names/counts at runtime.
- [x] Write gradient-reach and invariant tests: an update can change real `mu_post`; frozen weights, `mu_prior`, both latent log-variances and attention remain unchanged under deterministic evaluation.

**Code complete when:** the loader/wrapper and tests are written; no model construction or update has been executed. A classifier-only implementation is insufficient.

#### LP-06 — Write latent extraction and training scaling

- [x] Write train/validation mean/log-variance extraction with explicit identity/support metadata and row-aligned arrays. Preserve raw means as well as derived summaries.
- [x] Write checkpoint/config/support fingerprints and rejection of incompatible caches or before/after keys.
- [x] Write training-only hierarchical scaler moments with equal recording weights, the §5.2 floor/collapse rule, persistence and immutable application to later splits.
- [x] Write repeatability, weighted-moment and split-isolation tests. Test extraction remains callable only from the final evaluation path.
- [x] Frozen-fusion caching **omitted**, which this task explicitly permits: it is a speed optimisation whose equivalence check and runtime guard are the only things that would make it safe, and the extraction is a single forward pass per split rather than the run's bottleneck.

**Code complete when:** extraction/scaling functions and tests are authored; no latents, fitted scaler or cache have been produced.

#### LP-07 — Write preservation metrics and gates

- [x] Write deterministic validation-subset selection, including healthy-BG/adverse cases where available, and save its identities/support at runtime.
- [x] Write deterministic posterior-mean forecast MSE through the correct decoder/persistence/target path, using matched supported coefficients and recording reductions.
- [x] Write finite-value, healthy-only and posterior-delta saturation diagnostics and the declared MSE/saturation gates, including explicit zero-baseline-MSE handling.
- [x] Write matched-policy MC NLL/KL adapters with common random draws keyed to examples, plus an optional larger-draw convergence comparison for the user-run pipeline.
- [x] Write tests for unchanged-model identity, deliberate degradation and invalid/post-delivery endpoint exclusions.

**Code complete when:** gate decisions/reasons are fully represented in functions and test expectations; no candidate has been evaluated.

#### LP-08 — Write frozen baseline training

- [x] Write linear classifier fitting on pretrained posterior-mean bags, with class-balanced recording sampling or equivalent weights, never both.
- [x] Write validation AUROC stopping/selection, BCE/earlier-epoch ties, and the configured finite optimization budget.
- [x] Write validation-only balanced-accuracy threshold selection with a deterministic tie rule.
- [x] Write baseline classifier/scaler/threshold/history serialization and tests that VAE weights stay frozen and test inputs are inaccessible to fitting.

**Code complete when:** the entire baseline-fitting path is written and statically reviewed, with no classifier fitted by the agent.

#### LP-09 — Write tiny adaptation and selection

- [x] Write initialization from the original mean-head weights and the future frozen-baseline classifier artifact.
- [x] Write balanced late-bag BCE plus $0.1\,\mathcal L_{\rm keep}$, preserving supported three-hour segments of sampled eligible recordings without earlier outcome labels.
- [x] Write optimizer parameter groups, two learning rates, decay, clipping, ten-epoch ceiling, patience three and recording batching/accumulation; do not inherit pretraining optimizer/schedulers.
- [x] Write gate-first candidate eligibility and validation selection, retaining the frozen model as epoch zero when no eligible candidate improves selection.
- [x] Write history/rejection-reason logging and tests for loss reductions, teacher detachment, finite gradients, supervision scope and fallback behavior.

**Code complete when:** the adaptation/selection code and tests are authored; no optimization step has been executed.

#### LP-10 — Write persistence, resume and selection locking

- [x] Write pilot checkpoint serialization for adapted state, classifier/scaler, compatibility metadata, source identity, selected epoch and protocol/support fingerprints.
- [x] Write honest resume behavior with optimizer/RNG/stopping/stage state, or explicit new-run restart semantics where exact resume is unsupported.
- [x] Write base-model-compatible export excluding classifier keys and a strict-loader round-trip test definition.
- [x] Write selection-lock records fixing models, thresholds, gates, controls, projection procedure and seeds before test evaluation; enforce this dependency in code.
- [x] Write tests for reload equality, incompatible artifacts, source-checkpoint protection and completed-run overwrite protection.

**Code complete when:** persistence/locking code and tests exist; no real checkpoint has been loaded, changed or exported.

#### LP-11 — Write metrics, bootstrap and controls

- [x] Write recording-level AUROC/AP/prevalence/thresholded balanced accuracy and paired before/after comparisons on identical GUIDs.
- [x] Write outcome-stratified paired bootstrap with 1,000 draws, patient clustering if available, seeded resampling and explicit undefined-stratum handling.
- [x] Write the shuffled-label control starting from the original pretrained model, with its **own permuted-label baseline classifier**, independent validation-label permutation and true test labels at final evaluation. Never initialize from a true-label-fitted classifier.
- [x] Write healthy-BG restrictions, CS strata, acidosis/HIE contrasts and time/quality summaries without refitting per subgroup.
- [x] Write the conditional frozen `mu_prior` probe and its config switch, or explicitly exclude combined-branch-benefit claims when disabled.
- [x] Write known-answer metric/resampling tests, threshold-isolation tests and consistent-GUID-permutation tests. One control fit must never be called a permutation p-value.

**Code complete when:** evaluators and control-fitting functions are authored and their test definitions specify expected behavior; no metrics or control runs have been generated.

#### LP-12 — Write temporal and full-space geometry analyses

- [x] Write recording-bin summaries/scores, group counts and bootstrap bands with missing observations preserved.
- [x] Write paired-coverage early/late changes and subgroup descriptions with paired counts.
- [x] Write training-fitted class centroids, nearest-centroid evaluation and full standardized-space movement/covariance summaries.
- [x] Define effective rank, for example $\exp(-\sum_j p_j\log p_j)$ for normalized nonnegative covariance eigenvalues, with an explicit zero-variance convention.
- [x] Write one label-free training PCA shared by model versions, giving recordings equal weight, occupied bins equal within-recording shares and versions equal shares. Persist its variance/map and reuse without test-dependent fitting.
- [x] Write tests for bin membership, paired coverage, hierarchy, shared projections and train-only fitting.

**Code complete when:** analysis functions and tests are written; no PCA, centroid model or temporal result has been fitted/generated.

#### LP-13 — Write figure and report generators

- [x] Write the three §7.2 figure functions: common-axis before/after PCA, supervised-axis score distributions, and three-hour trajectories with counts and supervised-hour marking.
- [x] Write consistent class coloring/captions/variance labels and per-classifier score-scale disclosures. Do not interpret between-model logit scales as physiological motion.
- [x] Write seeded individual-trace selection and optional raw excerpts, preserving observed gaps/endpoints and avoiding favorable-example selection.
- [x] Write standalone PNG/PDF saving under the run directory; no external dashboard dependency.
- [x] Write the measured-report generator for provenance, exclusions, paired metrics/intervals, controls, preservation, temporal/subgroup findings, selection result, limitations and reproduction commands. Represent missing values honestly; never populate example clinical results as if measured.

**Code complete when:** generators/templates are written and data-to-caption/report wiring is reviewed; no figures or measured report have been rendered.

#### LP-14 — Write tests and the full smoke scenario

- [x] Complete the local behavior test suite specified in LP-02–LP-13 and focused integration tests for the existing loader/model/mask interfaces.
- [x] Write small fixture definitions/generator code with disjoint identities and both classes in evaluable splits. Artificial data/time/labels must remain explicitly nonclinical.
- [x] Write a full smoke scenario covering the internal extraction/baseline/adaptation/control/evaluation/report path, with tiny budgets and isolated outputs.
- [x] Write runner tests for no-argument IDE launch, direct-file/module/programmatic parity, non-repository working directories, override precedence, deep-copy behavior and no import-time work.
- [x] Write stage-order, failure-propagation, reload/resume, output-containment, source-integrity and report-only-without-retraining assertions.

**Code complete when:** test source and the smoke scenario are fully authored, with minimal logic tests isolated under `tests/logic/`. Leave execution until LP-16 for that subset only; smoke, model/integration, subprocess launch tests and fixture generation remain for the user on the other machine.

#### LP-15 — Wire every stage into the Run-button entry point

- [x] Finish `run.py` with editable `RUN_ARGS`, working function implementations, guarded import bootstrap, parser/resolver and `if __name__ == "__main__"` dispatch; remove all illustrative ellipses/stubs.
- [x] Wire every §11.2 stage, including `tests` and `smoke`, to the same dictionary-driven `main` API. Avoid requiring users to launch individual helper modules.
- [x] Write the complete `all` sequence with tests/smoke first, then production stages, selection locking before test, and analyses/report afterward.
- [x] Write per-stage resume/dependency logic and isolated smoke argument/output handling; avoid recursion, silent prerequisite retraining and production-argument mutation.
- [x] Review the direct-file root depth, argument precedence, stage wiring and absence of import-time side effects by reading source, without launching the runner.

**Code complete when:** all stages are reachable in the written dispatcher and the user can configure the entire flow through `RUN_ARGS`; execution remains unverified.

#### LP-16 — Minimal logic check and other-machine handoff

- [x] Review written code against §§1–10 and all tracker requirements, including the single-folder boundary and every critical temporal/gradient/split invariant.
- [x] After all code is written, run only `python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/logic -q` (or the repository environment's equivalent) for the tiny synthetic subset specified in §11.1. Fix logic failures and rerun affected tests only. If an existing lightweight dependency is unavailable, record that limitation and include the command in the handoff rather than installing the full environment or claiming success.
- [x] Update §12 and config/runner comments with the implemented keys, stage names, required paths and expected output locations. Include direct IDE Run instructions first and CLI alternatives second.
- [x] Write portable setup/transfer notes: required repository contents, Python/dependency expectations, target-machine paths to fill, device selection, stage order, resume instructions and expected outputs. Do not require access to the user's machine to finish this task.
- [x] Record each task's source-file evidence and the exact minimal logic command/result; distinguish those tests from the unexecuted integration/model suite and experiment. Do not claim verified model behavior or observed discrimination.
- [x] Document how the user can resume a stage, regenerate a report and supply traceback/logs for later fixes without silently changing the experiment protocol.
- [x] Provide a final handoff linking this document, `run.py`, configs and tests; state which tiny logic checks ran locally and list all remaining execution steps for the user on the other machine.

**Code complete when:** implementation and portable instructions are complete, the minimal logic results or specific limitation are disclosed, and the user has a single Run-button entry point for every remaining stage. Real artifacts/results and target-machine access are not prerequisites for this handoff.

### 11.6 Deferred extensions

Do not add more folds/seeds, larger unfreezing, supervised contrastive learning, learned MIL attention, temporal severity labels, monotonic penalties, onset detection, nonlinear-projection sweeps or calibrated risk prediction. Optional caching and the conditional prior probe fit within the stated tasks. Any broader follow-up belongs after the user's first completed experiment.

### 11.7 Source evidence and continuation log

Maintain a short entry after each meaningful code-writing step. Record file paths and inspected behavior. In LP-16 record the exact minimal logic command and actual result separately; the full integration/experiment field remains `USER_TO_RUN_ON_OTHER_MACHINE` unless the user later supplies results. Do not invent command outcomes.

| Date | Task ID | Written / reviewed by inspection | Source path / remaining work | Runtime verification |
|---|---|---|---|---|
| 2026-09-07 | Planning | Code-first checklist, minimal synthetic logic checks, portable IDE runner and separate execution-machine responsibilities specified | Start LP-01; no implementation created by this edit | USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-08 | LP-20 | **Multi-agent audit of the whole package against SS1-10, and the 47 findings it confirmed.** One was critical and silent: `pilot.yaml` carried **two `optim:` blocks** -- LP-19 added the 200/50 departure and, in the same commit, uncommented the default block whose header still says it is commented out -- so PyYAML kept the second and every run would have used `max_epochs: 10, patience: 3` while the file visibly read 200/50, discoverable only by diffing the two files afterwards. The block is re-commented; every value in it equalled `config.DEFAULTS`, so nothing else moved. Nine more were real defects. Two **empty-frame crashes** on admissible folds: `segment_means`/`recording_means` returned column-less frames, so a window with no anchor died as `KeyError: 'guid'` inside `stage_evaluate` -- after both extractions, the bootstrap and both preservation passes -- and eligibility constrains only the supervised window, so a fully eligible cohort can be unobserved in (2, 3] h; both reductions now carry their columns, and `window_scores` refuses by name rather than flowing an all-NaN contrast into the report as a measurement. A **NaN latent silently poisoned the frozen scaler** (NaN passes the collapse guard, the floor and every diagnostic, then makes the AUROC NaN, which the fit reads as "not better", so it retains epoch zero and finishes with nothing naming the cause) -- refused now, naming the coordinate. **Re-running `preflight` in a finished run directory** rewrote the recording table without the `eligible` column, and every consumer reads a missing column as no eligibility rule at all -- refused when `extract` has completed; and a NaN `eligible` read as True through `astype(bool)`, now `fillna(False)` at all three call sites. The **control's permutation crossed the eligibility boundary**, so it was fitted at a different class prevalence from the run it controls (and could abort the stage after the full epoch budget); it is now confined to the eligible population, verified invariant over 200 seeds. The **acidosis/HIE split of the paired early/late change** that SS7.3 requires was never computed and the paired table never persisted -- both added. The coverage table published **segment-start times under the name `last_observed_*`**, against the section SS4.2 is titled after -- renamed and computed over the usable rows the count beside it describes. The remaining 37 were minor and all applied: honesty in the report (the bootstrap's recorded failure reason was rendered as six "not measured" placeholders; the exposure limitation fired only when a record existed, i.e. never in the maximally unknown case; the movement subsection vanished silently; `reproduction_commands` omitted `--set`, printing a command the run itself refuses), hard-coded geometry in captions and headings, `supervised_bins` rounding a partly-outside bin as supervised, `group_bands` dropping the descriptive mean with the band it could not estimate and dropping unlabelled rows, `deterministic_outputs` silently filtering requested keys so `assert_invariants` could pass having compared nothing, an epoch-0 export named "adapted" with no `selected_epoch`, and two hot loops (`deduplicate_anchors`, 2.26 s -> 0.056 s on 20k rows with verdicts identical row for row; `segment_means`' per-group float64 cast). Inherited cohort-shaping loader filters are now refused rather than passed through unrecorded, and `provenance.dataset_build_mode` records the holdout/augmented mode SS4.1 asks for. **One finding was NOT fixed, on the operator's decision:** the shuffled-label control refits only the linear classifier on frozen pretrained latents and never reruns `fit_adaptation`, so it does not exercise the mean-head update or the epoch selection. The alternative was a second full adaptation plus a third test extraction, roughly doubling a run; the emitted `shuffled_label_note` now states the scope instead of implying a full refit. | `latent_pilot/{config,data,model,extract,train,evaluate,analyze,report,run}.py`, `latent_pilot/configs/pilot.yaml`, `latent_pilot/tests/logic/test_{cohort,scaler,temporal,baseline,metrics,report}.py`, `teb_vae/lag_attn_rws/tests/test_eval_launch.py` (the pilot runner added to `ENTRY_POINTS`, with the one declared `overrides`/`set_overrides` alias) | `pytest latent_pilot/tests --ignore=test_fixtures.py --ignore=test_smoke.py` plus `test_eval_launch.py` -> **466 passed**. The fixture-dependent `test_fixtures.py`/`test_smoke.py` and the experiment itself remain USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-19 | Progress reporting, after a run that spent minutes in silence. `tqdm` on the four loops that are actually long -- the per-split extraction forward, the preservation pass (`leave=False`, since it runs once per adaptation epoch and two hundred finished bars would bury the epoch lines), the baseline's step budget, and the adaptation's epochs with its within-epoch batches. The epoch bar carries a postfix of validation AUROC, training loss, the gate verdict and the epoch selected so far, so "where are we" and "is it working" are one line rather than two questions. Above them a stage counter: `run_pipeline` prints `pipeline stage 4/7: finetune` because it is what knows the ordered sequence, and `main` prints its own count only when it was handed more than one stage (the `--stage all` shape) -- it is called once per stage by the pipeline, and a `1/1` printed seven times underneath would say the opposite of the truth. Every stage also reports its wall clock, including on failure. The declared adaptation budget was raised to `max_epochs: 200`, `patience: 50` in `pilot.yaml` on the operator's instruction, marked there as a departure from the protocol's 10/3 | `latent_pilot/extract.py`, `latent_pilot/evaluate.py`, `latent_pilot/train.py`, `latent_pilot/run.py`, `latent_pilot/configs/pilot.yaml` | `--stage smoke` exit 0 with the bars and the `pipeline stage i/7` lines observed in the output; per-stage timings printed (preflight 7.9s, extract 2.2s, finetune 7.3s, evaluate 4.9s, report 1.8s on the fixtures). `pytest latent_pilot/tests -q --ignore=tests/test_smoke.py` -> **423 passed** |
| 2026-09-07 | LP-18 | **First end-to-end `smoke` run**, and the four defects between it and a finished pipeline. (1) `write_checkpoint` wrote the fit's configuration into a directory nobody had created -- `FileNotFoundError` on `fixture_fit.yaml`, after the shards and the statistics had already been built. (2) The fit then declared the *shipped* geometry over the *generated* shards: `tiny.yaml` carries the integer phase operator (44 `fhr_ph` + 10 `up_ph`, so `c_y` 80 / `c_u` 46) while `write_causal_cohort_shards` writes the legacy one (66 + 15, so 102 / 51) and takes no operator argument, and the trainer refuses the mismatch before the first batch. The widths and the operator name are now **read off the shard**, so the fixture fits whatever it was handed rather than pinning a triple that goes stale the day the cohort generator gains a switch. (3) The build's scratch lived in the **system temp**, which is why a run configured to write elsewhere printed `/tmp/latent_pilot_fixture_*` paths; it is now a `TemporaryDirectory` under the fixture root, swept on the way in and tolerant of a cleanup it cannot finish -- on Windows the fit leaves its own log sink open, and without that tolerance a fit that finished and wrote every artifact raised `PermissionError` on the way out and reported itself as a failure. (4) `stage_evaluate` could not write `band_table.parquet`: `analyze.group_bands` returned the grouping value verbatim, so the outcome grouping's `0`/`1` and the class grouping's `"acidosis"` landed in one object column and pyarrow refused it. The column is text now, which is how `report.py` already read it. **Separately, where a smoke run writes now follows the operator**: the outer run's `paths.run_root` is passed down as the one inherited leaf, so the run lands in `<run_root>/smoke/...` instead of inside the package, while the fixtures stay a per-checkout cache (re-rooting them would re-run the fit every time an operator changed where output goes) | `latent_pilot/tests/fixtures/generate.py`, `latent_pilot/analyze.py`, `latent_pilot/run.py` (`SMOKE_SUBDIR`), `latent_pilot/tests/logic/test_analysis.py`, `latent_pilot/tests/test_runner.py` | `--stage smoke` **exit 0**, whole pipeline: preflight, extract, baseline, finetune, control, evaluate, report, into the configured run root, with all four figures and `report.md` written. Fixture generation is a real fit and took **7.3 minutes** once. `pytest latent_pilot/tests -q --ignore=tests/test_smoke.py` -> **423 passed, 0 skipped** (the ten fixture-dependent tests now run), and `test_fixtures.py` 11 passed |
| 2026-09-07 | LP-17 | First execution of the `tests` stage on the execution machine, and the eight failures it produced. **Two were defects in the pilot itself.** `extract.py` called `max_forecast_shift` on `model` when it lives in `data` beside `latest_scored_seconds`, its only consumer, which took out every extraction path (`evaluate.py` already called it correctly). And the file-execution bootstrap in `run.py` added the repository root without removing **its own directory**, so `latent_pilot/train.py` shadowed the repository's top-level `train` package and `import train.graph_models_utils` failed as "train is not a package" -- several stages into a run, under exactly the Run button the convention exists to serve. The `_REPO_ROOT not in sys.path` guard made it worse rather than better: an inherited PYTHONPATH already carries the root further down the list, so the insert was skipped and the script directory kept position zero. **The other six were defects in the tests**, each asserting something stronger than the design: the shipped-configuration test required the production template's paths to be *unset*, which fails the `tests` stage of every run whose operator did what the file asks; the control test refused a recording table that carries test rows, which every production cohort does, rather than a *request* to permute them; the export test looked for the substring `linear`, which the net's own `target_adapter.linear.weight` matches; the determinism test compared arrays whose Monte Carlo columns are absent-as-NaN with a plain `array_equal`; the import test replaced PYTHONPATH instead of prepending to it, which on a remote interpreter drops the entry that resolves the editable in-repo `kymatio`; and the end-to-end plan test fed `recording_terms` its fixture's row order while `build_plans` sorts segments by start, so the teacher gap it asserts was zero measured a row mismatch. A **seventh of the same class** surfaced on the next run and is fixed the same way: `test_a_production_stage_without_paths_names_the_setting` launched the shipped template and expected `paths.checkpoint` in the refusal, which a filled-in template never produces -- it reaches the `extract`-needs-`preflight` refusal instead. It now writes its own `latent_pilot: {}` config, whose defaults leave the production paths unset. The run after that reached the **`smoke` stage, and found a second real defect**: the stage claims to need no data, but its fixtures are generated and git-ignored, and nothing on its path wrote them or named the generator -- so a fresh checkout got a six-line missing-input refusal ending in "'smoke' needs none of these". `stage_smoke` now writes them when they are absent (never when they are present: regenerating would spend the fit again and move the ground under a run that already read them), checks the result against the smoke configuration with the generator's own `manifest_matches`, and records `fixtures_generated`. `require_inputs` was split so the question "what is missing" can be asked by a caller that can fix the answer (`missing_inputs`), and its closing sentence no longer claims smoke needs nothing. Separately, `paths.run_root` was freed to point at any writable location, refused only when it *encloses* one of the run's own inputs | `latent_pilot/extract.py`, `latent_pilot/run.py`, `latent_pilot/config.py`, `latent_pilot/configs/pilot.yaml`, and the six test files; the bootstrap now has a regression test of its own (`test_the_file_form_does_not_shadow_the_repositorys_own_packages`) | `pytest latent_pilot/tests -q --ignore=tests/test_smoke.py` -> **410 passed, 10 skipped** in ~40 s on Windows/CPU. The smoke stage and everything past it remains USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-16 | Review, minimal logic check and handoff. **The review found one real gap and it was closed**: §7.3's coverage control asks for the same PCA coloured by ascertainment and coverage, and only the tabular half of that control existed -- so `figure_coverage_space` now draws the saved map coloured by blood-gas/Caesarean stratum and by how late each recording was still observed, on the pretrained version alone, because the question is not before-and-after but whether the separation tracks how these recordings were selected. Its bands are thirds of the last observed time, and a cohort too small to split says so rather than inventing them. The rest of the review confirmed by reading source: every new file lives under `latent_pilot/` with this document the single intentional exception; the only paths that read `paths.test_shards` are the evaluation stage's, every `split=\"test\"` sits after the lock, and preflight's read of the test shards is the metadata-only one §11.2 permits; the scaler, the class centroids and the projection are fitted on training rows only and `select_threshold` is called on validation logits only; `freeze_for_pilot` runs on every bundle any stage receives and `check_pilot_mode` before every optimizer step; the anchor clock is `epoch + trim_seconds + step_seconds x anchor` with the 49-minute known answer encoded in the temporal tests; both `torch.save` calls that could reach the source checkpoint are guarded; and no module derives a severity score, interpolates a missing time or imposes a monotonic constraint -- every textual hit for those words is a statement of the prohibition. §12 was rewritten as the execution guide it needs to be: what the machine needs, the five paths to fill, the Run button first and the CLI second, the artifact each stage actually writes under its real filename, resume / re-report / what-is-refused, what to keep when a stage fails **and why a protocol setting must not be edited to make one pass**, and a handoff that separates the 289 synthetic checks run here from the model tests, figures, fixtures, smoke run and every production stage that have been run nowhere. | `LATENT_CLASS_FINETUNING_PILOT.md` §12, `latent_pilot/report.py`, `latent_pilot/run.py`, `latent_pilot/analyze.py`, `latent_pilot/configs/{pilot,smoke}.yaml`, `latent_pilot/tests/logic/test_report.py`, `latent_pilot/tests/test_figures.py` | Minimal logic subset only, via the stand-in described in §11.1 because pytest is absent here: **289 checks, 0 failed**. The command for the execution machine is `python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/logic -q`. Everything else: USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-15 | Every stage wired into one dispatcher. The pipeline hands work between stages **through the run directory, not through memory**: the cohort table, the segment table, the latents, the scaler, the fits, the gate subset, the preservation readings, the projection and one `results.json` are all written, and each stage reads back what the one before it established -- so a stage-by-stage run and a single `all` produce the same directory, and a report assembled in a later process still names the checkpoint that produced it. **A checkpoint bundle is loaded fresh wherever it is needed and never cached across stages**: the fitting stage adapts the weights of the bundle it holds, so a shared one would hand `evaluate` an adapted model under the name `pretrained` and every paired comparison would be between two models that differ by nothing. `evaluate` locks the selection first and passes the lock's own permission into the test extraction as an argument, so the ordinary path to the held-out split runs through the check rather than beside it; a retry reuses the lock it already wrote rather than replacing it, because `open_run` has already refused any settings change and a second lock would be a record written after the split was available. Eligibility is attached **per split** -- judging every recording against one split's anchors would mark the rest ineligible for having no anchors in a pass that never read them -- so `extract` decides train/validation and `evaluate` decides test. Clustered resampling is passed only where a real patient mapping exists: the patient column falls back to the GUID, and handing it over unconditionally would report GUID-only grouping as patient grouping and suppress the disclosure that the interval may be too narrow. A cohort too small for the bootstrap is caught and recorded as a result about this fold rather than raised as a software failure, which is the distinction the report exists to keep. `smoke` resolves its own configuration and dispatches `run_pipeline`, which never runs `tests` or `smoke`, so it cannot recurse into itself or write into the production run whose sequence invoked it; `tests` runs the suite with the current interpreter from the repository root and ignores the smoke scenario, which is the next stage rather than a test of this one. Stage state is persisted as each stage finishes and on failure, so an interrupted run does not look, to the stage that resumes it, like a run that never fitted anything. | `latent_pilot/run.py`, `latent_pilot/data.py`. Checked statically only: no module-level call, every stage registered, every referenced module attribute exists, every call into the pilot modules binds against its real signature. No stage was executed here | USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-14 | Tests and the smoke scenario written. The fixture generator **rewrites two columns and generates nothing else**: the shards, their warm-up attributes, their channel plan and their statistics file are the repository's own tiny-shard generator's, and a second writer here would be a second description of a warm-up boundary. What it does rewrite is what the cohort generator answers a different question about -- **identities**, because the same shards are copied into three splits and without a prefix every GUID would sit in all three, leaving the pilot's disjointness assertion checking a property the fixture handed it for free; and **times**, because those segments sit about eleven hours before delivery, outside this pilot's window entirely, so an unrewritten fixture would filter every anchor away and report an empty cohort as a clean run. Each recording gets one segment inside the supervised hour and one in the early window, the last of them reaching past three hours so the boundary is exercised by a real segment; both clinical clocks move by the same offset rather than being left describing the old start. The checkpoint is a **real one-epoch fit** through `trainer.main`, copied out together with its `resolved_config.yaml`: a blob saved from a freshly constructed model would carry the same keys and prove none of them. The smoke scenario drives the stages **one at a time** through the same `main` an operator uses rather than through `all`, which begins with `tests` and `smoke` and would recurse into itself -- so the resume path is load-bearing in the scenario rather than tested beside it. Its ordering assertion is the one that matters: every test artifact must be younger than `selection_lock.json`, which is the mechanism turned into evidence on disk. The runner tests launch from a **foreign working directory with no `PYTHONPATH`** and require the three modes to refuse identically on the same bad argument -- agreeing on a refusal is the cheapest evidence they share one resolver -- and they distinguish the refusals by message, because a run-directory error rather than a config error is what proves a repository-root-relative path was resolved from somewhere else. The new logic file closes the LP-02 gap: schema, ranges, cross-checks, path resolution under a changed working directory, override precedence, `--set` YAML parsing, the deep-copy guarantee on `RUN_ARGS`, and an AST check that no module-level statement in the runner is a call. Two assertions state the dispatcher's contract before it exists -- every stage has a handler, no handler is registered under a non-stage name -- which is the order that makes them worth having. | `latent_pilot/tests/fixtures/generate.py`, `latent_pilot/tests/conftest.py`, `latent_pilot/tests/logic/test_settings.py`, `latent_pilot/tests/logic/test_report.py`, `latent_pilot/tests/test_fixtures.py`, `latent_pilot/tests/test_runner.py`, `latent_pilot/tests/test_smoke.py`, `latent_pilot/tests/test_figures.py`, `latent_pilot/tests/__init__.py`. No fixture was generated, no figure rendered and no stage run here | USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-13 | `report.py` written: three figures and one report, none of which holds a number of its own. **Both panels of figure 1 are drawn on the single saved projection and on limits computed over every point of every panel**, because two panels with their own limits would show a movement that was a change of axis range. Figure 2 is each model's *own* logit axis, and each panel says so on its axis label while the caption and the report carry the long form: the two models' scales are two fitted heads and a shift between them is not latent or physiological motion. Figure 3 shades the supervised hour, prints the recording count each band rests on at the foot of every bin, and leaves an unoccupied bin as a **gap** -- `nan` in the curve, a break in the line -- rather than interpolating a measurement that was never made; the individual traces come from `select_traces`, which reads GUIDs and class strata with a seeded generator and **cannot see a score**, which is the only defence against picking the traces that look convincing. Colours, captions and explained-variance labels come from one `CAPTIONS` table and the repository's own `group_colors`, so a pilot figure of the healthy cohort is the same blue as its training figure, and `_palette` extends that mapping to whatever a panel groups by so two groups never share one fallback grey. Rendering goes through `figures.render_figure` at the repository's DPI into `<run>/figures/`, in the format the new `figure_format` setting names; no dashboard, no new dependency. The report generator prints **`not measured`** for every value its record does not carry -- a missing section, a `nan`, an interval the estimator refused -- and carries the estimator's own reason where there is one, so a reader can separate a stage that never ran from a cohort too small to measure from a finding that came out inconclusive; a retained frozen model (epoch 0) is reported as the result it is, unknown pretraining exposure becomes a limitation rather than a clean-holdout claim, and the shuffled-label fit is never named a permutation p-value. The Markdown tables are rendered by a six-line helper rather than `DataFrame.to_markdown`, which needs a package this repository does not have. | `latent_pilot/report.py`, `latent_pilot/config.py` (`figure_format`), `latent_pilot/configs/pilot.yaml`, `latent_pilot/tests/logic/test_report.py`. Figures are rendered only by the LP-14 smoke scenario; no figure and no measured report was produced here | USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-12 | `analyze.py` written. Trajectory summaries carry **absence through** rather than filling it: a recording with no retained anchor in a bin has no row there, the count printed under each bin is what its band actually rests on rather than the cohort size, and a bin too small to resample reports its count and no band. One **frozen** final-hour head scores every bin -- refitting per bin would make each bin's score a different quantity -- and the bins outside the supervised hour are marked on the row, because the loss was defined on the last hour and a score two hours out is an application of that head outside its own window. The bands are the repository's own `stats.bootstrap_ci` over recordings, so a band here means what a band means elsewhere in the tree. The paired early/late comparison keeps only recordings observed in **both** windows and reduces both by the same anchors-then-segments-then-recordings order, so the difference is not an artefact of two aggregation rules; the record carries the sentence the result supports and the two it does not, because "consistent with changing outcome-associated signal" and "evidence of worsening" are one edit apart in a report. Geometry is computed in **full latent space**, never on projection coordinates: training-fitted class centroids and a nearest-centroid score read through no fitted head, so linear discrimination improving while the class centres stay put is a statement this package can make. Effective rank is $\exp(-\sum_j p_j\log p_j)$ over the **non-negative** eigenvalues -- an eigendecomposition's small negatives are numerical, not directions -- and the zero-variance convention is **0.0**, explicitly: nothing varying is no direction carrying anything, not one direction carrying everything. The projection is one weighted PCA fitted label-free on the training bin vectors of **every version at once**, at $1/V$ per version, $1/N$ per recording and $1/B_i$ per occupied bin, so the map describes the cohort rather than whoever was recorded longest; its axes are sign-fixed deterministically, because an eigendecomposition may return either sign and a panel that flipped between two runs would read as movement. It refuses any frame carrying a split other than training, and it is persisted so every later panel is drawn on the axes this one fitted. **Dedup, touching LP-08 and LP-09:** the eligible-and-labelled filter existed in three copies (`build_bags`, `build_plans`, and the analyses would have been a fourth); it is now `data.eligible_anchors`, because three copies of a filter are three chances for one of them to keep a recording the others dropped | `latent_pilot/analyze.py`, `latent_pilot/data.py`, `latent_pilot/train.py`, `latent_pilot/tests/logic/test_analysis.py`; next unfinished task LP-13 (figure and report generators). No PCA, centroid model or temporal result was fitted here | Logic subset: 202 checks passed, 0 failed (29 new), run without pytest via a throwaway harness. Nothing in LP-12 needs a model, so all of its checks are in that subset. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-11 | Held-out metrics, the paired bootstrap and both controls written. The bootstrap is **paired by construction**: one draw of units scores every model, because two independent bootstraps give two intervals whose overlap says nothing about the paired change. It resamples **patients where a mapping exists and GUIDs otherwise**, and GUID-only grouping is *disclosed in the record* rather than left to be inferred -- repeated recordings of one delivery are not independent, and an interval that assumed they were would be too narrow. Strata are the units' own outcome signatures, so a patient contributing both a healthy and an adverse recording forms a stratum of its own instead of being forced into one by majority; every stratum keeps its size, so a draw cannot silently become one on which AUROC is undefined, and a draw that does is counted and excluded rather than dropped. Percentile intervals with `MIN_UNITS = 3`, matching `lag_attn.eval.stats`, so "significant" means here what it means elsewhere in the repository; the point estimate is the metric on the **full** sample, because the draws estimate the spread and not the value. Average precision is always reported beside the prevalence that sets its chance level. The strata -- healthy-BG controls, CS/no-CS, acidosis and HIE against the same healthy controls -- are produced **together, in a fixed order, for every model at once**, which is what stops the best one being promoted afterwards, and a stratum too small to estimate discrimination reports counts and `nan` rather than a number. The shuffled-label control permutes the **recording table** rather than threading a label mapping through every stage: that is what makes it the same code path as the real run, since `build_bags`, `build_plans` and the validation scoring all read outcomes from there and none of them needs a control-aware branch that could drift from the branch it controls. Each split is permuted on its own draw, the test split is refused outright, and `fit_control_baseline` **requires the permutation record** rather than asserting `labels_permuted` on a table it cannot inspect -- a fit claiming a shuffle without the draw that produced it would be an unverifiable line in a report. The frozen `mu_prior` probe fits its own scaler on `mu_prior`'s own training anchors, because standardizing one latent by another's spread would make the comparison a comparison of scalings; switched off, `control_disclosure` withdraws the combined-branch claim instead of leaving it implied, and one shuffled fit is never called a permutation p-value | `latent_pilot/evaluate.py`, `latent_pilot/train.py`, `latent_pilot/tests/logic/test_metrics.py`, `latent_pilot/tests/test_controls.py`; next unfinished task LP-12 (temporal and full-space geometry analyses). No metric, interval or control fit was produced here | Logic subset: 173 checks passed, 0 failed (26 new), run without pytest via a throwaway harness. `test_controls.py` fits classifiers and was **not** run here. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-10 | Run-directory bookkeeping written into `config.py`, the pilot checkpoint into `train.py`. **Resume is stage-level, and that is a stated decision rather than an omission:** exact mid-fit resume would have to persist the optimizer's moments, the sampler's stream and the RNG state every epoch and would still not reproduce a run bit for bit across a device change, so it would be a promise this package could not keep; the fits are at most ten epochs over two small heads, restarting one is cheaper than the machinery that avoids it, and `open_run` says so where a reader will look. `open_run` is the one place the three cases are decided -- a new run writes its protocol and stage state; a resume reads the protocol back and **refuses a changed settings digest**, naming the dotted paths that moved, because continuing a run under different choices leaves a directory whose protocol describes a run that never happened; a re-entry without resume may run only `RERUNNABLE_STAGES`, which is what lets §12's own instruction -- regenerate the report from saved artifacts -- work without letting a finished fit be overwritten in place. `require_completed` refuses a stage whose input was never produced, so a `report` request never silently refits its own subject. The selection lock is written **once** and cannot be re-locked: a second lock is by definition a choice made after the held-out split was available, which is the one thing it exists to prevent; `require_selection_locked` returns `True` so it reads as `allow_test=require_selection_locked(run_dir)`, putting the check where the permission is granted. The pilot checkpoint carries the adapted mean heads **only** -- writing the whole net out as though it had been trained would misdescribe a 2,112-parameter change -- plus the classifier with its scaler as buffers, the threshold, the selected epoch, the source checkpoint's digest and the support fingerprint; `apply_adapted` checks the digest and the parameter names before writing anything, so an adaptation of one checkpoint cannot be dropped into another and a geometry change surfaces there rather than as a shape error deep in a forward. `export_base_checkpoint` writes the net's own state dict with **no classifier key** anywhere, stamped with the source's `model_class`/`model_kwargs`/`hyper_parameters` so `check_model_class` and `load_checkpoint_strict` both find what they expect, and copies the resolved configuration in beside it where `resolved_config_for` looks. `_refuse_source_path` guards both writers: the pretrained checkpoint and its configuration are opened read-only for the whole pilot | `latent_pilot/config.py`, `latent_pilot/train.py`, `latent_pilot/tests/logic/test_run_state.py`, `latent_pilot/tests/test_persistence.py`; next unfinished task LP-11 (held-out metrics, bootstrap and controls). No checkpoint was loaded, changed or exported here | Logic subset: 147 checks passed, 0 failed (15 new), run without pytest via a throwaway harness. `test_persistence.py` builds the net and writes tensors and was **not** run here. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-09 | Adaptation half of `train.py` written. The unit of supervision is a **recording**, which the ordinary loader cannot be asked for, so `RecordingSource` holds the dataset and a `(guid, epoch)` index and collates one recording's segments **in the plan's order** -- a sampler could not promise that, and a bag whose segment weights landed on the wrong rows would be wrong without being wrong-shaped. **Support is not recomputed during training:** `build_plans` reads it off the pretrained extraction, which already applied the contributing rule, the window, the post-delivery check and both duplication rules, so the anchors the student is supervised and held at are the anchors every table in the run is built from -- and a pandas deduplication stays out of the inner loop of a fit. **The teacher is the cached extraction rather than a second forward through a frozen copy:** $\mu^q_0$ is a deterministic function of the inputs and the extraction is that function at exactly these anchors, so a deepcopy would recompute numbers the run already has; the identity is *verified* rather than assumed -- before the first update the student **is** the teacher, so `recording_terms` returns the largest student-minus-teacher gap and `_verify_teacher` refuses a cache the first forward does not reproduce, which is free because that forward happens anyway. The latent is gathered at the anchors' own **step indices** straight off $\mu^q$'s time axis, so nothing depends on how the forward laid out its anchor axis. `epoch_batches` is the **sampling** arm of the either/or, so the classification loss is a plain mean and `balanced_bce` is deliberately not applied on top of it; batches are whole blocks cut from a fresh permutation per class rather than slices of a concatenated stream, because a slice straddling two shuffles can present one recording twice in a batch of eight -- the cost is that a pool the block does not divide leaves a remainder for the next reshuffle, which is the right way round when the loss is defined per recording. A class smaller than `recordings_per_class` shrinks the batch instead of repeating a recording. Each recording's loss is backpropagated as it is computed and the step is taken once the batch completes, so memory is a function of one recording; the optimizer is built here from `parameter_groups` with two learning rates and **no scheduler at all**, and `check_pilot_mode` runs before every step rather than once. Selection is gates-first: a failing candidate is ineligible however good its AUROC, its reason is recorded, and epoch zero -- the frozen model with the baseline classifier -- is always eligible, so a run where nothing improves reports the pretrained model rather than an arbitrary changed one; `_reference_gate` states that rather than comparing the baseline against itself, which would say nothing at a zero baseline. **Defect fixed across LP-06/LP-07 as well:** none of the pass loops moved its batch to the model's device, which would have failed every GPU run; `model.to_device` now routes all three through the task's own `transfer_batch_to_device` hook, the same one the evaluation pipeline uses | `latent_pilot/train.py`, `latent_pilot/model.py`, `latent_pilot/extract.py`, `latent_pilot/evaluate.py`, `latent_pilot/tests/logic/test_adaptation.py`, `latent_pilot/tests/test_adaptation_fit.py`; next unfinished task LP-10 (persistence, resume and selection locking). No optimization step was executed here | Logic subset: 132 checks passed, 0 failed (18 new), run without pytest via a throwaway harness. `test_adaptation_fit.py` runs real forwards and optimizer steps and was **not** run here. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-08 | Baseline half of `train.py` written. The fit receives **no model**: bags are pooled posterior means already extracted from the pretrained checkpoint, so nothing in `fit_baseline` can reach a VAE parameter and "the baseline is the frozen model" is a property of the signature rather than a claim beside it. Bags come from `data.recording_bags` -- the same reduction the adaptation's forward path will use -- over the extraction's **full** array, because `row` indexes that array and a gathered matrix would be addressed by the ungathered positions (the LP-06 test made exactly that call and is corrected here). Balance is the class-weight arm of the protocol's either/or and only that arm: a full batch of cached vectors has nothing to sample, so `balanced_bce` carries it, and the docstring states that a fit drawing class-balanced batches must use the plain mean instead rather than both. **Stopping and selection are deliberately two rules**: §6 stops on epochs without an improved validation AUROC while selection is AUROC, then lower BCE, then the earlier step -- folded into one, a fit whose ranking had saturated would spend its whole budget chasing a cross-entropy that is only ever a tie-break, so patience counts AUROC alone and is checked after selection so the step that ends the run can still be selected. Validation is read at natural prevalence, which is what makes its BCE the protocol's tie-break rather than the objective's own number. The threshold maximises balanced accuracy over `roc_curve`'s own candidates, dropping the leading infinite one that no finite score can meet, and ties resolve to the **lower** threshold -- towards sensitivity -- rather than to whichever candidate the curve listed first. AUROC, the cross-entropy, balanced accuracy and the threshold scan live in `evaluate.py`, where LP-11 will read them, so the selection metric and the reported metric are one definition; AUROC is sklearn's, because a local rank formula agrees with it only if it handles ties by mid-rank. The classifier saves and loads **separately from the base model**, its scaler travelling as buffers inside it, and the pretrained checkpoint is never written to. `_labels_for` is the seam the shuffled-label control will fit through: per GUID, refusing a mapping that omits a recording | `latent_pilot/train.py`, `latent_pilot/evaluate.py`, `latent_pilot/tests/logic/test_baseline.py`, `latent_pilot/tests/test_baseline_fit.py`, `latent_pilot/tests/test_extraction.py` (scaler call corrected); next unfinished task LP-09 (the tiny adaptation and checkpoint selection). No classifier was fitted here | Logic subset: 114 checks passed, 0 failed (24 new), run without pytest via a throwaway harness. `test_baseline_fit.py` constructs a classifier and takes gradient steps and was **not** run here. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-07 | Preservation half of `evaluate.py` written. The gated forecast is decoded from $\mu^q$ rather than from a draw of $z^q$: the forward's own `mu_full` reads the sampled `z_post`, so two readings of one model would differ and a gate could be passed or failed by the noise between them. The decode is the model's own shared decoder at the anchors the forward returned, carrying the forward's `persistence` tensor where the checkpoint was built with the residual, against `_build_forecast_target` -- so the per-channel forecast clock is applied by the code that owns it rather than by a copy. Support is the objective's own: one `forecast_mask` at the checkpoint's `coverage_floor`, `contributing_anchors` on that same mask -- the composition `data.contributing_support` performs, unrolled only because the mask itself is scored against here -- then the window, the post-delivery endpoint check and both duplication rules, in the order the latent tables apply them, so the gate and the analysis describe one anchor set. The MSE divides each anchor's masked block by the objective's fixed $H \cdot C_{\rm keep}$ rather than by its own surviving element count, so the number drifts with forecast error and not with mask density; the reduction is anchors -> segments -> recordings, so a densely covered recording cannot decide a gate. `mse_base` is decoded beside it as an invariant: `mu_prior` is frozen, so it must be bit-identical before and after. **Deviation, with reason:** healthy-only drift is measured and *reported* rather than gated -- §6 declares one forecast tolerance, and applying it a second time to the smaller, noisier stratum would reject candidates the declared protocol accepts; it is raised as a warning at the same tolerance, so a candidate that preserved the adverse group's forecast by spending the healthy group's cannot pass quietly. A zero baseline MSE falls back to a named rule (`ZERO_BASELINE_RULE`) instead of dividing, and a non-finite *baseline* raises rather than failing the candidate -- that is a broken measurement, not a property of the adaptation. Every reading carries a `support_digest` over its retained `(guid, epoch, anchor)` keys, and `gate_decision` refuses a pair whose digests differ, so a change of population can never be reported as a change of forecast error. Monte Carlo draws go through the evaluation package's own `mc_predictive_block` with a generator seeded from a blake2b of the batch's `(guid, epoch)` pairs -- keyed to the examples, not counted per batch, so the frozen and the adapted model see identical noise even across processes -- and are skipped by default, because eight draws per branch per batch is the expensive part of something that runs once per candidate epoch while the marginal NLL is wanted twice in a whole run. One new setting, `gates.subset_recordings` (24), fixes how many validation recordings the gate is measured on; the draw is round-robin across adverse / healthy-BG / healthy-no-BG after a seeded permutation of GUID-sorted pools, so a small subset is not one class's subset, and it is saved rather than recomputed | `latent_pilot/evaluate.py`, `latent_pilot/config.py`, `latent_pilot/configs/{pilot,smoke}.yaml`, `latent_pilot/tests/logic/test_gates.py`, `latent_pilot/tests/test_preservation.py`; next unfinished task LP-08 (frozen linear baseline fitting). No forecast was decoded and no candidate was gated here | Logic subset: 90 checks passed, 0 failed (25 new), run without pytest via a throwaway harness. `test_preservation.py` builds the real net and was **not** run here. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-06 | `extract.py` written. The pass runs under `no_grad` in evaluation mode and needs no seed: it keeps the posterior and prior **means and log-variances**, all deterministic, so a repeated read is bit-identical (`z` is never used). Raw per-anchor values are stored, not only reductions, so every bag, bin, window and geometry summary is recomputable later. Anchors outside the preserved window are filtered **before** their vectors are gathered and reported as a count -- the window is a definition, not an exclusion -- while support, the post-delivery endpoint and both duplication rules are marked on the row in that order, so the first reason an anchor failed is the one reported. The support fingerprint carries the checkpoint digest, the resolved geometry, the coverage floor, the forecast clock, the dense `(0, 1)` decoding and the window; `assert_same_keys` refuses a before/after pair that is not row-for-row identical rather than joining around the difference, which would drop exactly the rows that matter. The scaler's moments are hierarchical -- anchors, segments, recordings, each recording weighted equally -- so a densely covered recording cannot set the cohort's scale; scales are floored at max(1e-3, 0.1 x median positive) and a constant latent raises `LatentCollapse` rather than being scaled against a floor. The fitted arrays are marked read-only, so an in-place edit downstream fails loudly instead of silently rescaling every later number, and `fit_scaler` refuses any frame carrying a split other than `train`. `check_split_allowed` keeps the test split unreadable outside the evaluation stage, enforced in code because 'we only looked at it at the end' is not demonstrable from a run directory afterwards | `latent_pilot/extract.py`, `latent_pilot/tests/logic/test_scaler.py`, `latent_pilot/tests/test_extraction.py`; next unfinished task LP-07 (preservation metrics and candidate gates). No latents, fitted scaler or cache were produced here | Logic subset: 65 checks passed, 0 failed (15 new). The extraction tests build the real net and were **not** run here. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-05 | `model.py` written. Reconstruction goes through `eval/probe.py`'s `load_task` with `TRF_CFS_BINDING`, so the class guard, the `model_kwargs`/`hyper_parameters` refusals and the checked `load_checkpoint_strict` are the evaluation pipeline's rather than a second copy; the **task** is kept because the five-argument forward is assembled through its own input seam, and `forward_inputs` asserts that seam resolved the dense (0, 1) geometry rather than the training tiling. Geometry is read from the checkpoint's `model_kwargs` for declared values and from the rebuilt net for resolved ones, and logged; no width, horizon, trim or clock is written into this package. `statistics_record` **refuses** a stats file built at a different `trim_minutes` (the loader only warns, and a warning inside a multi-hour extraction is not a guard) and *discloses* rather than refuses a file repointed away from the checkpoint's own, which a fold split legitimately requires. The freeze switches every parameter off and the `delta_mu_head` parameters back on, calls `eval()` (dropout off, autograd untouched), and derives the trainable count from the loaded checkpoint instead of assuming 2,112; `check_pilot_mode` is cheap enough to call per step and catches both a stray `train()` and a leaked `requires_grad`. `parameter_groups` is an explicit allowlist, and the classifier's scaler is held as **buffers** so it travels with the state dict and cannot be optimised. `deterministic_outputs` saves and restores the global RNG state around its seeding, so a gate called mid-fit cannot silently advance the training draw | `latent_pilot/model.py`, `latent_pilot/tests/test_model_contract.py`; next unfinished task LP-06 (latent extraction and the fixed training scaler). Import-checked only: the module imports cleanly and `LatentClassifier` constructs, but no checkpoint was loaded and no model was built | Gradient/invariant tests are written and **not run here** — they build the real net, which the local subset excludes, and pytest is unavailable locally. USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-04 | Temporal half of `data.py` written. **Deviation, with reason:** the trim offset is taken through the loader's own `decimated_trim_steps` rather than as the $60m$ the protocol states, because the loader discards a whole number of *decimated steps*; the two are identical at the shipped `trim_minutes: 1.0` (240 raw samples, 15 steps, 60 s) and the 49-minute worked case is unchanged, so nothing about comparability moves -- they diverge only for a trim whose sample count is not a multiple of the decimation, where $60m$ would place every anchor of that dataset up to 3 s early. The segment span for the coarse filter is likewise derived from the checkpoint's `sequence_length` and trim (300 steps + 2x15 trimmed = 1320 s) rather than from the 22 minutes today's shards hold. Support is the objective's own: `model.scored_weight` then `raw_masks.forecast_mask` at the checkpoint's `coverage_floor`, then `contributing_anchors` -- one policy, no second definition, and the inherited forecast-availability exclusions are recorded as a limitation. Windows and bins are half-open at the early edge and closed at the late one to a 1e-9 hour tolerance, so bins tile without sharing an anchor. Post-delivery is checked on the furthest scored endpoint $t + H + \max_c s_c$, so an advancing forecast clock changes the verdict. Deduplication marks rather than deletes, keeping the anchor with more in-segment history and then the earlier segment start. Reductions are strictly anchors -> segments -> recordings everywhere, so a dense segment cannot outvote a sparse one and duplicated anchors cannot multiply a recording's supervised weight; unoccupied bins are absent rather than filled | `latent_pilot/data.py`, `latent_pilot/tests/logic/test_temporal.py`; next unfinished task LP-05 (strict model loading and gradient allowlist). Source review confirms no severity label, no ordinal target, no time interpolation and no monotonic constraint anywhere in the module | Logic subset: 50 checks passed, 0 failed, run without pytest (unavailable locally) via a throwaway harness; full suite and experiment remain USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-03 | Cohort half of `data.py` written. Loader assembly copies the checkpoint's own resolved config and changes four things only (shard lists, statistics, `load_fields` extended by `target`/`cs_label`/`bg_label`/`time_from_labor_onset`/`second_stage_onset`, coarse `epoch_min`), refusing a run whose config never loaded `weight`/`guid`/`epoch` or a coefficient stream, forcing `label: null` and single-process loading through `config_schema.force_single_process_loader`; both shard lists are repointed so no pretraining path survives. One identifying pass per split recovers the class code only through `labels.clinical_class_code` (the target/weight ratio, with zero meaning *no class*), plus subgroup, CS/BG flags and both clinical clocks; the clinical fields are batch identity and are never forward inputs. Recording consolidation **excludes** a GUID whose segments disagree about class or metadata rather than voting, and counts every exclusion reason. Split disjointness is checked by GUID and, where a patient map exists, by patient; absent, GUID-only grouping is disclosed in the record. Exposure keeps `unknown` (no list supplied) distinct from `disjoint` (list supplied, no overlap), and only the latter sets `clean_holdout_supported`. Coverage emits a row for every canonical subgroup including the empty ones, and `require_both_classes` refuses a split that cannot estimate discrimination. Batch-field helpers are local copies rather than imports of `eval/probe.py`, which reaches the model through its binding and would pull it into the light logic subset | `latent_pilot/data.py`, `latent_pilot/tests/logic/test_cohort.py`; next unfinished task LP-04 (anchor timestamps, support masks, late bags and trajectory bins). Logic tests written for conflicting labels, zero/invalid targets, split overlap, recording grouping, patient grouping, exposure provenance, class availability, empty strata, loader assembly and serialisation | Logic subset: 19 passed, 0 failed, run without pytest (unavailable locally) via a throwaway harness that calls each `test_*` directly; full suite and experiment remain USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-02 | Strict nested settings schema written as the `DEFAULTS` tree, with unknown-key refusal at every depth, nullable/list type tables, positive/non-negative/min/max range tables and cross-checks (windows nest, bins divide the preserved window, early window outside the supervised hour, `mc.large_draws >= mc.draws`, shard lists disjoint after path resolution). Repository-root path normalisation, and a run destination that may be any writable location but is refused when it encloses one of the run's own inputs. `require_inputs` demands production paths only for the stages that read them, so `tests`/`smoke`/`report` resolve without clinical data. Run-ID (timestamp + settings digest), protocol record (settings, run args, git revision and dirty flag, tolerance disclaimer) and initial stage state, all construction-only. Argument resolver shared by the Run button, `python -m`, direct execution and `main(**RUN_ARGS)`, with per-key precedence (explicit CLI > non-`None` dictionary > YAML > defaults), `argparse.BooleanOptionalAction` and `--set a.b=c` YAML-parsed structured overrides, deep-copied inputs and one validator for both sources. Reuses `teb_vae.lag_attn.config.load_config` and `_deep_merge` (the same private binding the two sibling eval packages already import) rather than restating the merge rule | `latent_pilot/config.py`, `latent_pilot/configs/pilot.yaml`, `latent_pilot/configs/smoke.yaml`, `latent_pilot/run.py`; next unfinished task LP-03 (cohort/label/provenance manifest). Not exercised locally beyond a syntax check: the resolver logic is covered by the `tests/logic` subset written in LP-14 and run once in LP-16 | USER_TO_RUN_ON_OTHER_MACHINE |
| 2026-09-07 | LP-01 | Package skeleton written with one documented responsibility per module; reuse points recorded by inspection of `eval/binding.py` (`TRF_CFS_BINDING`, `GEOMETRY_KEYS`), `lag_attn_rws/eval/run.py` (`load_task`, `resolve_device`, strict-load refusals), `train/graph_models_utils.py` (`check_model_class`, `load_checkpoint_strict` returning `None`), `lag_attn/nets/heads.py` (`PosteriorHead.delta_mu_head` is a `ModuleList[Linear(fuse_out, d_z//num_heads)]`, `fuse_out = max(2*group, 16)`), `lag_attn_cfs/nets/causal_inputs.py` (forward output keys incl. `anchor_index`/`anchor_valid`/`delta_mu_sat_frac`), `lag_attn_cfs/nets/causal_feature_target.py` (`_build_forecast_target`, `scored_weight`, `target_forecast_shift`), `lag_attn_rws/nets/raw_masks.py` (`forecast_mask`, `contributing_anchors`, `kl_mask`), `lag_attn_rws/nets/losses.py`, `lag_attn/eval/labels.py` (`clinical_class_code`, `CLASS_NAMES`, `CANONICAL_SUBGROUPS`), `lag_attn/eval/{stats,figures,report,masks,numerics}.py` via `lag_attn_cfs/eval/_reuse.py`, `lag_attn_cfs/eval/collect.py` (`PER_ANCHOR_KEY`, `check_per_anchor_key`), `lag_attn_cfs/eval/cohort.py` (`add_time_bins` bins segment-start `epoch` — the gap this pilot fills), `hdf5_dataset/hdf5_dataset.py` (`CombinedHDF5Dataset` filters, `decimated_trim_steps`, `RAW_SAMPLING_HZ=4`/`DECIMATION=16` ⇒ 4 s step), `train/data_module.py` (`GraphDataModule`), `trainer.py` (Run-button/bootstrap pattern), and the environment, which this repository deliberately does not pin — no dependency manifest, no lock file, no interpreter pin, and no `install_requires` in the root `setup.py`; the pilot adds no dependency and creates no root dependency file, and the packages its import chain reaches are recorded in `latent_pilot/__init__.py` without versions | `teb_vae/lag_attn_transformer_cfs/latent_pilot/`; next unfinished task LP-02 (config schema, templates and argument resolution) | USER_TO_RUN_ON_OTHER_MACHINE |

Decision/continuation entry:

```text
Date / task:
Code or documentation written:
Source interfaces and behavior reviewed:
Decision/deviation and reason:
Effect on protocol or comparability:
Next unfinished coding task:
Minimal local logic tests: command/result, or not run with reason
Full integration/experiment verification: USER_TO_RUN_ON_OTHER_MACHINE
```

Mark implementation complete only after all required code-writing tasks are done. Do not leave coding tasks blocked merely because the user has not run them or supplied production paths. Do not restart completed source work in later sessions unless new information requires it.

## 12. User execution guide — on the other machine after coding is complete

**Owner: the user, on the machine agents cannot access. This section does not authorize agent-side experiment execution.** Everything below describes code that is now written; none of it has been run against real data, and a finished coding checklist is not evidence that any test passed or that any separation improved.

### 12.1 What you need on that machine

| Requirement | Detail |
|---|---|
| Repository | This revision, checked out whole. The pilot imports the surrounding packages (`teb_vae.lag_attn*`, `hdf5_dataset`, `train`, `utils`, `scripts`) and will not run from the `latent_pilot/` folder alone. |
| Python environment | Any environment in which the packages listed in `latent_pilot/__init__.py` import: `torch`, `lightning`, `numpy`, `pandas`, `pyarrow`, `h5py`, `pyyaml`, `loguru`, `matplotlib`, plus `scipy`/`scikit-learn` by way of the pipeline this package imports. **The repository pins nothing** — no dependency manifest, no lock file, no interpreter pin — so there is no environment to install from and no version this pilot requires. `pytest` is needed for the `tests` stage and for nothing else. |
| Working directory | Irrelevant. Every path resolves against the repository root, and direct execution of `run.py` moves there itself. |
| Device | One GPU or none. `device: null` chooses `cuda:0` when available, else CPU; `RUN_ARGS["device"]` or `--device` overrides. The trainable set is two small heads. |
| Data | The pretrained checkpoint **together with the `resolved_config.yaml` beside it**, the statistics file that checkpoint was trained under, and the fold-1 train/validation/test shard lists. A checkpoint copied away from its resolved config cannot be rebuilt and the loader says so by name. |

### 12.2 Fill in five paths

Open `latent_pilot/configs/pilot.yaml` and replace the five `REPOINT_ME` entries under `paths`: `checkpoint`, `statistics`, `train_shards`, `val_shards`, `test_shards`. Supply the optional `provenance` entries if you have them — `pretraining_guids`, `selection_guids`, `patient_map`, `statistics_population`. Absent, exposure is recorded as **unknown**, which is not the same statement as "disjoint" and which the report prints as a limitation.

The alternative is the `overrides` block of `RUN_ARGS` in `run.py`, which takes the same nested keys and wins over the YAML. Both are recorded in the run's protocol, so neither hides what the run read. Every other setting already carries the declared protocol value; the commented block at the bottom of the YAML shows what those values are, and uncommenting one is a departure from the protocol that should be recorded.

Nothing in that file can reconfigure the checkpoint. Architecture, channel contract, trim, forecast clock, horizon, latent width and anchor stride are read from the checkpoint's own saved kwargs, and a pilot key for any of them does not exist and is refused as unknown.

### 12.3 Run it — the IDE Run button first

1. Open `teb_vae/lag_attn_transformer_cfs/latent_pilot/run.py` in the intended environment. Edit `RUN_ARGS` at the bottom of the file. Hit Run. No CLI flags, no launcher, no working-directory setting.
2. `RUN_ARGS["stage"] = "tests"` → the written test suite, with the current interpreter, from the repository root. Needs no production data. The end-to-end smoke scenario is excluded here because the next stage is it.
3. `"smoke"` → the whole pipeline on small **artificial** fixtures, in its own run subtree. Generate them once first:
   `python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate`
   It writes into `latent_pilot/tests/fixtures/generated/` (git-ignored) and runs one real one-epoch fit to produce a checkpoint, so expect minutes. A finished smoke run proves the stages connect and the artifacts round-trip. It is evidence about nothing else: the identities, times and labels are invented, and no number it produces may be quoted.
4. `"preflight"` → the real cohort: identities, class codes, split disjointness, patient grouping, exposure, class presence, coverage, and the checkpoint's geometry. Fits nothing. Resolve everything it reports before going on.
5. `"all"` → the whole ordered sequence in one run directory:
   `tests → smoke → preflight → extract → baseline → finetune → control → evaluate → report`.
   Or select the production stages one at a time in that order, reusing the same run directory (see §12.5).

CLI equivalents, for a shell or a job script:

```bash
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage tests
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage smoke
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage preflight
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run \
    --config teb_vae/lag_attn_transformer_cfs/latent_pilot/configs/pilot.yaml --stage all
python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage evaluate \
    --run-dir teb_vae/lag_attn_transformer_cfs/latent_pilot/runs/fold_1/seed_42/<run_id> --resume
python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests -q
```

Direct file execution uses the same dictionary when no CLI argument is supplied, and an explicit CLI value overrides the corresponding entry key by key:

```bash
python teb_vae/lag_attn_transformer_cfs/latent_pilot/run.py
python teb_vae/lag_attn_transformer_cfs/latent_pilot/run.py --stage preflight
```

`--set` supplies structured overrides, parsed as YAML: `--set optim.max_epochs=3`, `--set paths.train_shards=[a.hdf5,b.hdf5]`.

### 12.4 What each stage writes

Under `<run_root>/<fold>/seed_<seed>/<run_id>/`, and nowhere else. `run_root` defaults to `latent_pilot/runs` (the tree `.gitignore` covers) but may be any writable location, relative to the repository root or absolute; it is refused only if it contains one of the run's own inputs — the checkpoint, the statistics file or a shard.

| Stage | Artifacts |
|---|---|
| any | `protocol.yaml` (settings, run arguments, git revision, tolerances), `stage_state.json` (completed, failed, `selection_locked`) |
| `preflight` | `segments.parquet`, `recordings.parquet`, `split_manifest.csv`, `coverage.csv`, `preflight.json` (checkpoint identity and digest, statistics record, exposure, grouping, exclusions) |
| `extract` | `train_pretrained_*` and `val_pretrained_*` (`latents.npz`, `latent_index.parquet`, `latents.json`), `latent_scaler.json`, and the manifest/coverage rewritten with eligibility |
| `baseline` | `baseline_classifier.pt`, `baseline_fit.json` |
| `finetune` | `gate_subset.json`, `gate_reference_preservation.json`, `pilot_checkpoint.pt`, `adapted_model/adapted_model.ckpt` (base-model-compatible, classifier keys excluded, with the checkpoint's resolved config copied beside it so the export loads on its own), `train_adapted_*`, `val_adapted_*` |
| `control` | `control_classifier.pt`, `control_fit.json`, and `prior_probe_*` when the probe is enabled |
| `evaluate` | `selection_lock.json`, `test_pretrained_*`, `test_adapted_*`, `pretrained_preservation.json`, `adapted_preservation.json`, `projection.json`, `metrics.csv`, `per_recording_test.parquet`, `per_recording_bin.parquet`, `trajectory_bands.parquet`, `figure_bags.npz`, `results.json` |
| `report` | `figures/figure1_latent_space`, `figures/figure1b_latent_space_by_coverage`, `figures/figure2_supervised_axis`, `figures/figure3_trajectories` (in `figure_format`, `pdf` by default), and `report.md` |

Read `report.md` first. Every value it prints as **`not measured`** is one the run did not establish — not one that came out at zero.

### 12.5 Resuming, re-reporting, and what is refused

* **A new run** leaves `run_dir` unset and gets a fresh directory.
* **Continuing one** names it and sets `resume: true`. Completed stages are skipped. Resume is **stage-level**: an interrupted fit restarts from its own beginning, not from its last epoch, and that is stated rather than implied.
* **Re-reporting** names the directory *without* `resume`. Only `tests`, `smoke`, `preflight` and `report` may run that way; they rewrite presentation and touch no fitted artifact. `report` re-renders the figures and `report.md` from `results.json` and needs neither the shards nor the checkpoint, so it works on a machine where the data is no longer mounted.
* **Refused, deliberately:** resuming under changed settings (the protocol record would describe a run that never happened); re-running a fitting stage in place in a finished directory (later artifacts were selected against it); asking for a stage whose prerequisite never ran (`report` never silently trains, `evaluate` never silently extracts); and reading the test split before the selection lock exists.
* A second experiment is a **new run directory**, not an edit of an old one.

### 12.6 When something fails

The stage that failed is recorded in `stage_state.json` with its reason, and the sequence stops there. For a fix, keep: the failing stage name, the traceback, the console log, `protocol.yaml`, `stage_state.json`, and — where they exist — `preflight.json` and `coverage.csv`. Those say what was read and under which settings.

Do not repair a failure by changing a protocol setting. Windows, eligibility rules, gates, learning rates, batch composition, epochs and seeds are declared before extraction; changing one to make a stage pass turns a fixed protocol into a swept one, and the run's own settings digest will no longer match the directory it is continuing. If a setting genuinely has to change, start a new run and record the reason.

A failed validation pilot is still a result. So is a run that selects epoch zero and reports the frozen model.

### 12.7 Handoff: what has been run, and by whom

| Artifact | Where |
|---|---|
| Protocol, tracker and this guide | `teb_vae/lag_attn_transformer_cfs/LATENT_CLASS_FINETUNING_PILOT.md` |
| Entry point | `latent_pilot/run.py` (`RUN_ARGS`, `main`, `STAGE_HANDLERS`) |
| Configuration | `latent_pilot/configs/pilot.yaml`, `latent_pilot/configs/smoke.yaml` |
| Implementation | `latent_pilot/{config,data,model,extract,train,evaluate,analyze,report}.py` |
| Tests | `latent_pilot/tests/logic/` (synthetic subset) and `latent_pilot/tests/` (the rest) |
| Fixture generator | `latent_pilot/tests/fixtures/generate.py` |

**Run on the coding machine:** the synthetic logic subset only, and **`pytest` is not installed there**, so it was exercised by importing each test module and calling every `test_*` function directly through a throwaway stand-in for `pytest.raises` / `approx` / `mark.parametrize` / fixtures — **289 checks, 0 failed**. Run the real command yourself:

```bash
python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/logic -q
```

**Not run anywhere yet:** the model and integration tests, the figure rendering, the fixture generation, the smoke scenario, the runner launch tests, and every production stage. No checkpoint has been loaded, no latent extracted, no classifier fitted, no figure drawn and no report rendered. Nothing in this repository contains a measured result of this experiment.

**Remaining steps, all yours:** generate the fixtures; run `tests`; run `smoke`; run `preflight` on the real fold and read its counts and exposure; run the production sequence; read `report.md`, the metrics and the figures. Report failures with the material in §12.6.

The eventual report should separate software/runtime failures, insufficient cohort support, and an inconclusive or negative scientific finding. A finished coding checklist does not imply that tests passed or that healthy/adverse latent separation improved.
