# Evaluation rigour beats architecture for latent CTG classifiers

For a downstream classifier on VAE latents of intrapartum CTG, the evidence points one way. **Keep the encoder frozen at first. Pool within each segment with gated attention. Run a small causal sequence model over the observed segment tokens with time and stage embeddings. Train with plain BCE (or a cumulative-link ordinal head). Get calibrated risk and an FPR-capped alarm post hoc, on patient-grouped validation data, never on test.** Architecture choices come second. The published CTG models cluster at **AUROC 0.75–0.85 and roughly 43–58% sensitivity at 10–15% FPR** whatever the backbone. The 2026 gains come from pretraining, not from new heads ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf); [Fridman & Ben Shachar 2026](https://arxiv.org/abs/2601.06149)). Evaluation choices, on the other hand, move headline numbers by far more than any model change. Dropping intermediate-pH cases, evaluating only the last window before delivery, splitting segments rather than patients, and picking the FPR threshold naively can each inflate results. In one ICU study the framing choice alone moved AUROC from 0.906 to 0.756 on the same data ([Lauritsen 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593052/)). The naive "empirical FPR ≤ α" threshold breaks its cap about half the time ([Tong, Feng & Li 2018](https://www.science.org/doi/10.1126/sciadv.aao1659)). Several of the requested components have no CTG precedent at all: attention-MIL, ordinal 3-class losses, explicit conditioning on time since labour onset, and modelling missing segments. For these the recommendations borrow from ICU early-warning, long-tailed learning and clinical-prediction methodology, and should be treated as well-grounded defaults, not proven CTG results. The report is written to TRIPOD+AI, and the Python stack it needs is almost entirely scikit-learn, scipy and plain PyTorch already in the project venv.

## Published CTG models plateau near 0.8 AUROC, and outcome definitions drive the numbers

Cord-artery pH is the dominant label, and how the literature cuts it decides which results can be compared. The public benchmarks are small and curated. **CTU-UHB has 552 recordings, selected from 9,164 for signal quality and gas availability, with only 46 caesareans and stage 2 capped at 30 minutes** ([PhysioNet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)). It contains about 40 cases with pH<7.05 (7%), 65 intermediate (7.05–7.15) and 447 normal ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/)). SPaM is a 300-case, 80:20 case-control set with no intermediate cases at all, so AUCs on it are optimistic by design ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)). The large private cohorts use three-level pH outcomes:

- **Paris APHP (DeepCTG 2.0), 27,662 cases:** pH>7.20 normal, 7.05–7.20 moderate (12.5%), ≤7.05 severe. The model is evaluated as two binary tasks, "moderate+severe vs normal" and "severe vs rest", not as an ordinal model ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)).
- **Melbourne MHW-pH, 9,887 recordings:** 1.1% compromised ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/)).
- **UPenn, 10,182 tracings:** reported at four cut-offs. AUROC falls steadily from **0.85 at pH<7.05 (1.3% prevalence) to 0.75 at pH<7.20 (20.9%)**, so the choice of threshold alone dominates the headline number ([McCoy 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499302/)).
- **Oxford:** adds a clinical composite of "severe compromise": stillbirth, neonatal death, encephalopathy, seizures, or resuscitation followed by more than 48 h in NICU. It is 452/51,449 births. The label targets HIE-type harm directly, not acidaemia ([Asfaw 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294944/)).

HIE is a hard target. At 1–3 per 1,000 births, about 30,000 deliveries yield only about 100 cases ([O'Sullivan 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8576107/)). Any HIE classifier built on a cohort of hundreds to a few thousand patients is effectively an acidaemia or composite classifier, and should be reported that way.

The intermediate pH band is where results are most often inflated. DeepCTG 2.0 notes that an earlier Oxford AUC of 0.82 "was obtained on a subset of the dataset excluding the cases of moderate acidemia". Excluding intermediates at test time makes results incomparable ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)). Excluding them from *training* is defensible but rarely matters. Mendis dropped pH 7.05–7.15 from training only and kept those cases in test as normals ([Mendis 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11144251/)). A later test found **no statistically significant change** from excluding them ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/)). The rule is to always evaluate on the full test population, intermediates included.

| Model (cohort) | Evaluation | Headline result |
|---|---|---|
| OxSys 1.5 (Oxford, 22,790) | vs clinical practice | Severe-compromise sensitivity 43.3% vs 38.0%, FPR 14.4% vs 16.3% ([Georgieva 2017](https://obgyn.onlinelibrary.wiley.com/doi/10.1111/aogs.13136)) |
| MCNN on CTU-UHB | TPR at FPR 5/10/15/20% | 33/48/58/65% ([Mendis 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11144251/)) |
| FHR-LINet (CTU-UHB) | TPR at FPR 5/10/15/20% | 27.5/45.0/56.5/65.0%, about 25% earlier detection than MCNN ([Mendis 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11144251/)) |
| InceptionTime (UPenn) | pH<7.05, last 60 min | AUROC 0.85, sens 79%, spec 78%, PPV 5%; CTU-UHB zero-shot 0.72 ([McCoy 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499302/)) |
| DeepCTG 2.0 CNN (5 centres) | leave-one-centre-out | Severe 0.74–0.83; moderate+severe 0.65–0.85 ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)) |
| ResNet cross-database | MHW→CTU / CTU→MHW | 0.81 / 0.65 ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/)) |
| Early-labour CNN (Oxford, first 20 min) | severe compromise | AUC 0.68, 20% sens at 95% spec ([Asfaw 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294944/)) |
| PatchTST masked SSL (preprint) | CTU-UHB, pH<7.15 | AUC 0.83 (0.853 uncomplicated vaginal) ([Fridman 2026](https://arxiv.org/abs/2601.06149)) |

Three patterns in this table shape the design of a latent classifier:

- **Supervised architecture has saturated.** InceptionTime beat CNN, LSTM and Transformer rivals at UPenn ([McCoy 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499302/)). DeepCTG's CNN beat LSTMs and slightly beat a CNN+Transformer.
- **Pretraining is where the gains are.** Pretraining DeepCTG's conv layers to predict expert CTG features lifted AUC from 0.72 to 0.75 (moderate+severe) and from 0.78 to 0.82 (severe) ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)). This supports the VAE-latent strategy. The one strong SSL number, Fridman's 0.83, comes with a caveat: it rests on a **55-recording test split with 11 positives**, and the unlabelled pretraining pool included the CTU-UHB recordings themselves ([Fridman 2026](https://arxiv.org/html/2601.06149v1)). PRISM-CTG, a multi-view CTG foundation model, claims parity with models trained on much larger labelled sets but reports no AUCs in its abstract ([arXiv 2605.02917](https://arxiv.org/abs/2605.02917)).
- **Generalising across sites is the real ceiling.** AUC moves by 0.16 depending on which database trains and which tests.

Operating points are reported at fixed FPR, and 15% is the de facto clinical anchor, because clinicians run at roughly 14–16% FPR ([Georgieva 2017](https://obgyn.onlinelibrary.wiley.com/doi/10.1111/aogs.13136); [Asfaw 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294944/)). A project that caps FPR at 30% is working at twice clinical practice. It should also report TPR at 5/10/15/20% FPR, or its numbers cannot be compared with MCNN, FHR-LINet or OxSys.

The literature also documents the failure modes a latent classifier inherits:

- **Weak labels.** Labelling a whole recording from one birth outcome creates "predominantly noisy labels", especially for acute events where "the CTG may only change during the event" ([O'Sullivan 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8576107/)).
- **Treatment paradox.** Fridman found that false positives often matched CTGs judged concerning on clinical review, and AUC was higher on uncomplicated vaginal deliveries. This fits a picture where timely intervention converts "abnormal CTG" into "normal pH" ([Fridman 2026](https://arxiv.org/abs/2601.06149)). The mitigation is to stratify evaluation by delivery mode, and to treat "caesarean or instrumental delivery for presumed fetal compromise" as a separate outcome, not to merge it silently into either class.
- **Informative signal loss.** In CTU-UHB, 59% of normal but only 32% of compromised cases have less than 20% loss. The FIGO <20%-loss rule would exclude 41% of normals and 68% of compromised cases ([Mendis 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11144251/)). Low signal quality is a leading cause of DeepCTG's false negatives ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)).
- **Unreliable toco (UC).** FHR-only beat FHR+UC across databases, 0.71 vs 0.67 ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/)). In the Paris data the reverse held and FHR+UC was best.

## Weak labels demand horizon-aware targets and a single ordinal ranking

The outcome is measured once, at delivery, but a continuous classifier scores segments hours earlier. Copying the patient label to every segment is the weakest possible supervision. Asfaw's early-labour model reached only AUC 0.68, against 0.75–0.85 for last-60-minute models ([Asfaw 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294944/)). Early segments of acidaemic babies often look normal. The CTG convention sidesteps this by labelling only the final 60 minutes ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/); [McCoy 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499302/)). That is not possible for whole-labour prediction.

The borrowed alternatives, from simplest to most principled:

1. **Hard-horizon targets.** Positive only within H≈60 min of delivery.
2. **Temporal label smoothing.** The target decays smoothly with time-to-event. It improved early event prediction in the clinically relevant high-precision region on ICU data ([Yèche et al. 2023](https://arxiv.org/abs/2208.13764)).
3. **MIL with the patient label on the bag.** This formalises "at least one segment is pathological" ([Ilse et al. 2018](https://arxiv.org/abs/1802.04712)).
4. **A discrete-time logistic-hazard head.** Hazard h_j = σ(φ_j(x)) and survival S(τ_j) = Π(1−h_k), fit by the censored likelihood ([Kvamme & Borgan 2019](https://arxiv.org/abs/1910.06724)), in the style of Dynamic-DeepHit ([Lee et al. 2020](https://ieeexplore.ieee.org/document/8681104/)).

A workable combined recipe is a patient-level BCE on the pooled or last-state output plus λ≈0.3–1 times a segment-level BCE. Positive-patient segments are weighted by exp(−Δ/τ), where Δ is time to delivery, and all negative-patient segments get full weight. Two cautions apply. First, **time to delivery is a target-construction variable only.** It is unknown at inference and must never be an input. Second, a hazard framing only approximates a latent "onset of compromise", because the event is observed at birth. No CTG paper compares these label schemes head to head. That comparison is an open, publishable ablation.

For the 3-class target (normal / intermediate / adverse), **no CTG or clinical study was found that uses an ordinal loss**. The rational default is a cumulative, rank-consistent head:

- **CORAL.** Shared weights, K−1 ordered biases, P(Y>r_k) = σ(g(x)+b_k) ([Cao, Mirjalili & Raschka 2020](https://arxiv.org/abs/1901.07884)).
- **Proportional-odds cumulative link.** Structurally the same ([McCullagh 1980](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x)).

The practical advantage is decisive. Under CORAL or proportional odds, every cumulative probability is monotone in one latent score g(x). So **P(Y ≥ adverse) and P(Y ≥ intermediate) share one ranking and the same AUROC**. One model then yields both DeepCTG-style binary tasks and a single consistent alarm score. The alternatives break this property. CORN relaxes the weight sharing via conditional chain-rule probabilities ([Shi, Cao & Raschka 2023](https://arxiv.org/abs/2111.08851)), and SORD uses distance-based soft labels on a softmax ([Díaz & Marathe 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html)). Both give each cut-point its own ranking. Squared EMD equals the ranked probability score for one-hot targets, which makes it a strictly proper score for ordinal evaluation ([Hou, Yu & Samaras 2016](https://arxiv.org/abs/1611.05916)). With only three classes, the differences between heads will be small. Choose CORAL for its single score and minimal parameters.

## Freeze first, fine-tune second, co-train last

With hundreds to a few thousand patients and rare positives, the adaptation ladder should be climbed only as far as validation evidence requires.

**Frozen encoder plus a small head is the baseline.** On 12-lead ECG, linear evaluation of self-supervised features came within about 0.5% of supervised performance ([Mehari & Strodthoff 2022](https://www.sciencedirect.com/science/article/pii/S0010482521009082)).

**Reconstruction-trained features tend not to be linearly separable.** MAE features are notably weaker under a linear probe than under fine-tuning, and tuning only the last one or two blocks recovers most of the gap ([He et al. 2022](https://arxiv.org/abs/2111.06377)). Expect an MLP or attention head over VAE latents to beat a pure linear probe. Probing several encoder depths is a cheap ablation, since multi-layer linear probes approach fine-tuning ([Head2Toe](https://arxiv.org/abs/2201.03529)).

**When frozen features plateau, use LP-FT: train the head to convergence, then fine-tune.** Full fine-tuning with a randomly initialised head distorts pretrained features. LP-FT gave about 1% better in-distribution and about 10% better out-of-distribution accuracy than full fine-tuning ([Kumar et al. 2022](https://arxiv.org/abs/2202.10054)). Given CTG's severe cross-site shift, the out-of-distribution result matters most. Cheaper variants:

- Tune only the top encoder block ([surgical fine-tuning](https://arxiv.org/abs/2210.11466)).
- Use LoRA on the attention projections ([Hu et al. 2022](https://arxiv.org/abs/2106.09685)).
- Use discriminative learning rates, with the encoder at 10–100× below the head ([ULMFiT](https://arxiv.org/abs/1801.06146)).

No CTG study compares linear probe, LP-FT and LoRA. Fridman's SSL model was fine-tuned end-to-end with no probe ablation ([Fridman 2026](https://arxiv.org/html/2601.06149v1)).

**Joint VAE+classifier co-training is the fragile end of the ladder.** Kingma's M2 model needs an ad hoc classification term weighted at α = 0.1·N, because q(y|x) otherwise gets no labelled signal ([Kingma et al. 2014](https://arxiv.org/abs/1406.5298)). CCVAE notes that α has no probabilistic basis ([Joy et al. 2021](https://arxiv.org/pdf/2006.10102)). Prediction-constrained work finds that semi-supervised generative objectives give poor predictors unless the prediction term is heavily up-weighted ([Hope, Hughes et al.](https://arxiv.org/pdf/2012.06718)). Stripping the generative machinery from semi-supervised VAEs did not hurt text classification: the gains come from the discriminative path ([ACL Insights 2021](https://aclanthology.org/2021.insights-1.19.pdf)). The core tension is structural. The ELBO's KL term pulls label-relevant information toward the prior, while a supervised head fights posterior collapse only in the dimensions it uses ([Bowman et al. 2016](https://arxiv.org/abs/1511.06349)).

If co-training is attempted anyway:

- Tune the classifier weight λ on validation.
- Use a free-bits KL floor.
- Monitor per-dimension active units.
- Consider PCGrad or uncertainty weighting for gradient conflict ([Yu et al. 2020](https://arxiv.org/abs/2001.06782); [Kendall et al. 2018](https://arxiv.org/abs/1705.07115)).

A stop-gradient variant, where the classifier reads sg(z) through a trainable adapter, preserves the generative model exactly. It is the cleanest way to keep the VAE's reconstructions and KL "surprise" signals valid for interpretation.

## A two-level hierarchy absorbs gaps, irregular time and missing covariates

The data structure is up to 22 fixed-length segments per labour, some absent and non-contiguous, each carrying per-timestep latents [T, D]. It maps onto a two-level model.

**Within a segment, use gated-attention pooling.** The attention weights are a_k ∝ exp(wᵀ(tanh(Vh_k) ⊙ sigm(Uh_k))). It outperformed other MIL methods on 50–150 bags and yields per-instance weights ([Ilse et al. 2018](https://arxiv.org/abs/1802.04712)). A single learnable query, i.e. PMA with k=1, is the equivalent Set-Transformer form ([Lee et al. 2019](https://arxiv.org/abs/1810.00825)). Concatenate mean and max pooling of μ as a robustness baseline. Global average pooling was part of why ResNet generalised best across CTG databases ([Mendis 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250915/)). MILLET's conjunctive pooling, ŷ = Σ a_t f(h_t), multiplies attention by per-timestep logits and gives more faithful per-timestep evidence than raw attention, without losing accuracy on 85 UCR datasets ([Early et al. 2024](https://arxiv.org/abs/2311.10049)). That makes it the better choice if the per-timestep explanation plots are a deliverable.

**Across segments, use a one- or two-layer causal transformer or a GRU over segment tokens.** A 1–2 layer transformer uses d≈64–128 and a causal mask. It emits a risk r_i at every observed segment for the continuous display, and the last-state risk is the online patient score. With at most 22 tokens, sequence length is irrelevant, so S4 or Mamba buy nothing ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)). Over-parameterising the aggregator is the main overfitting risk. For a retrospective patient label, pool the r_i with attention or log-sum-exp.

The fixed MIL aggregators are baselines:

- mean
- max
- noisy-OR, 1−Π(1−p_i)
- LSE, (1/r)·log mean exp(r·s_i)

FHR-LINet's "recording positive if any window positive" rule is max-pooling. Its authors flag its equal weighting of windows as a limitation, because "fetal compromise may occur at a specific point" ([Mendis 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11144251/)). **Avoid noisy-OR.** With variable segment counts it mechanically inflates risk for patients with more observed segments. This concern is analytic, not empirically tested in CTG.

**Missing segments are dropped, not imputed.** Absent tokens go through `src_key_padding_mask`, and each token carries its time context:

- A learnable linear-plus-sinusoid embedding of minutes since labour onset (mTAN/Time2Vec form, φ(t)₀ = ω₀t+α₀, φ(t)ᵢ = sin(ωᵢt+αᵢ)) ([Shukla & Marlin 2021](https://arxiv.org/abs/2101.10318)).
- log(1+Δt) since the previous observed segment.
- A stage embedding.
- The fraction of the segment with raw-signal loss.

This is the SeFT idea: treat observations as a set of timestamped tokens, which is natively robust to irregular sampling ([Horn et al. 2020](https://arxiv.org/abs/1909.12064)). In a GRU, GRU-D's learned decay on the gap δ is the cheap equivalent. GRU-D also showed that missingness patterns are informative in ICU data ([Che et al. 2018](https://arxiv.org/abs/1606.01865)), and simple missingness indicators improved RNN diagnosis ([Lipton et al. 2016](https://arxiv.org/abs/1606.04130)).

Heavier continuous-time models are unlikely to pay off at this scale. ContiFormer ([Chen et al. 2023](https://arxiv.org/abs/2402.10635)), Latent ODEs ([Rubanova et al. 2019](https://arxiv.org/abs/1907.03907)) and Raindrop ([Zhang et al. 2022](https://arxiv.org/abs/2110.05357)) were benchmarked on cohorts of thousands to tens of thousands of patients.

**Stage and elapsed time are full context, not optional covariates.** The literature handles stage only by building separate models. MCNN reached AUC 0.65 on the last 60 minutes of stage 1 and 0.71 on the last 30 minutes of stage 2 ([O'Sullivan 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8576107/)). Stacking stage models is another route ([Petrozziello 2019](https://ora.ox.ac.uk/objects/uuid:7046540a-cd61-42b6-acf9-4e8de34a5dde)). Some studies simply exclude second stage ([Asfaw 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10294944/)). **No study was found that conditions on time since labour onset**, so per-token time and stage embeddings are novel in this domain. Report them as such.

The embeddings also address a documented failure: bradycardia during pushing is a DeepCTG false-negative category ([Ben M'Barek 2025](https://lepennec.perso.math.cnrs.fr/Reprint/Health/2024-CBM-BMJMKSCLPS.pdf)). The INFANT trial's decision support, tested in about 47,000 women, failed partly because it ignored labour duration and progress ([O'Sullivan 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8576107/)). That is an argument for conditioning on labour context, not against it.

**Clinical covariates add a little, and their missingness is a trap.** Clinical features lifted DeepCTG 1.5 from AUC 0.74 to 0.77, and meconium-stained fluid consistently raises risk ([Menzhulina 2025](https://pubmed.ncbi.nlm.nih.gov/39893788/)). Expect +0.02–0.05. The standard design:

- Build a covariate vector of imputed values plus a binary missingness mask.
- Pass it through a small MLP.
- Inject it either as an extra token in the patient transformer, as FiLM on segment embeddings ([Perez et al. 2018](https://arxiv.org/abs/1709.07871)), or, least prone to overfitting, by late fusion before the head.
- Apply per-covariate and whole-block dropout during training (ModDrop) so a CTG-only path survives ([Neverova et al. 2016](https://arxiv.org/abs/1501.00102)). This also gives a clean CTG-only vs CTG+covariate ablation from one model.

Missing indicators are double-edged. In simulation they improve prediction under informative missingness. But when missingness depends on the outcome and is not allowed at deployment, they worsen Brier score and calibration-in-the-large ([Sisk et al. 2023](https://arxiv.org/abs/2206.12295)). Models that exploit missingness patterns can also fail when the patterns shift between sites ([arXiv 2406.16484](https://arxiv.org/html/2406.16484)). Maternal temperature is the textbook case: it is measured because of concern. Validate with and without the indicators. Admit only covariates available at prediction time. Delivery mode, intervention flags and retrospective labour duration leak the outcome.

**Auxiliary VAE signals belong in the attention scorer, not in the pooled representation.** Per-timestep KL, ½Σ(μ²+σ²−log σ²−1), and posterior variance are plausible "surprise/quality" cues, but there is no evidence they help downstream classification. Generative likelihoods are known to misrank out-of-distribution data ([Nalisnick et al. 2019](https://arxiv.org/abs/1810.09136)). Attention weights are contested as explanations ([Jain & Wallace 2019](https://arxiv.org/abs/1902.10186); [Wiegreffe & Pinter 2019](https://arxiv.org/abs/1908.04626)). A defensible compromise:

- Compute attention as a_t = f(μ_t, KL_t, log σ²_t, mask_t) while pooling only μ_t, so low-quality stretches can be down-weighted without contaminating the representation.
- Feed summaries (entropy, max) of the encoder attention maps rather than raw T×T maps.
- Use μ at inference and sampled z as training augmentation.
- Validate each input by ablation with patient bootstrap.

## Plain BCE plus post-hoc calibration and thresholds beats imbalance tricks

The strongest methodological evidence in the notes concerns imbalance handling, and it cuts against common deep-learning practice. In clinical prediction simulations, **undersampling, oversampling and SMOTE gave no AUROC benefit but pushed calibration intercepts to ≤ −4.5 at 1% prevalence, i.e. severe risk overestimation**, and produced negative net benefit at higher thresholds ([van den Goorbergh et al. 2022](https://academic.oup.com/jamia/article/29/9/1525/6605096)). The Monte Carlo follow-up found uncorrected ML models "consistently had equal or better calibration", and recalibration could not always repair the damage ([Carriero et al. 2025](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.10320)). A 1.8-million-patient real-data study found that cost-sensitive learning and resampling left AUROC and AUPRC "unchanged regardless of the method" while inflating predicted risk by up to 62.8% ([Roesler et al. 2026, preprint](https://www.medrxiv.org/content/10.64898/2026.03.04.26347634v1.full)). In that study isotonic recalibration largely repaired the damage. The two findings conflict in degree. The safe reading is that avoiding the distortion beats repairing it.

The mechanism is simple for a binary head. A positive weight w shifts the logit intercept by about log w. Logit adjustment ([Menon et al. 2021](https://arxiv.org/abs/2007.07314)) and balanced softmax ([Ren et al. 2020](https://arxiv.org/abs/2007.10740)) reduce to a constant bias for a sigmoid. Neither can change AUROC. Each amounts to moving the threshold, which post-hoc selection does anyway.

The other imbalance tools, briefly:

- **Focal loss** (−α(1−p_t)^γ log p_t) was designed for dense detection. Its default α=0.25 *down-weights* positives ([Lin et al. 2017](https://arxiv.org/abs/1708.02002); [torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html)). Its calibration benefit applies to overconfident, large, overparameterised nets ([Mukhoti et al. 2020](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)), so on small clinical data it must be checked, not assumed.
- **LDAM-DRW and asymmetric loss** come from many-class long-tail vision and multi-label tasks. With 2–3 classes, their benefit collapses into a bias shift ([Cao et al. 2019](https://arxiv.org/abs/1906.07413); [Ridnik et al. 2021](https://arxiv.org/abs/2009.14119)).
- **Decoupling is the transferable lesson.** Natural sampling learns the best representations, and rebalancing, if done at all, belongs on the head ([Kang et al. 2020](https://arxiv.org/abs/1910.09217)). A frozen-encoder classifier already is "the head".
- **Symmetric label smoothing** sets a floor of ε/2 on predicted risk. That is a real bias when the base rate is 2%.

The clustering problem is more important than class imbalance. Weighting each segment by 1/(number of segments for that patient), or sampling patients first, stops long labours from dominating the loss.

Directly optimising the operating region is a legitimate but unproven ablation. LibAUC provides AUC-M and one-way pAUC losses, where pAUC over FPR ≤ β matches an FPR-capped alarm ([Yuan et al. 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html); [Zhu et al. 2022](https://proceedings.mlr.press/v162/zhu22g.html)). The docs recommend pretraining with CE and then fine-tuning at a lower learning rate ([LibAUC docs](https://docs.libauc.org/api/libauc.losses.html)). The reported gains are the authors' own, on vision and molecular tasks. No independent clinical replication or CTG use was found. These losses need a positive-enriched sampler: at 2% prevalence and batch size 64, about 27% of batches contain no positive. Their outputs are not probabilities, so recalibration is mandatory.

**Calibrate with Platt scaling (binary) or temperature scaling, fit on validation predictions only.** Isotonic calibration needs on the order of 100–200 positives (a heuristic, not a sourced rule). Beta calibration handles S-shaped miscalibration ([Kull et al. 2017](https://proceedings.mlr.press/v54/kull17a.html)). Vector scaling suits a 3-class softmax ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)). For CORAL, refit the scale on g(x) and the K−1 offsets, which preserves rank consistency.

If any reweighting was used, first remove it analytically with the prior-shift correction, logit p_corr = logit p − log w for pos_weight w ([Saerens et al. 2002](https://doi.org/10.1162/089976602753284446)), then recalibrate.

Report calibration with:

- calibration intercept and slope, on the clinical hierarchy of mean, weak and moderate calibration ([Van Calster et al. 2019](https://doi.org/10.1186/s12916-019-1466-7));
- a smoothed calibration curve with ICI/E50/E90 ([Austin & Steyerberg 2019](https://doi.org/10.1002/sim.8281));
- log loss;
- the *scaled* Brier score. At π=0.02 the uncertainty term is only 0.0196, so raw Brier values look deceptively small.

Binned ECE is bin-dependent and biased ([Nixon et al. 2019](https://arxiv.org/abs/1904.01685)). At rare prevalence almost every prediction falls in the lowest equal-width bin, so if ECE is reported at all, use equal-mass bins or smooth ECE ([Błasiok & Nakkiran](https://arxiv.org/abs/2309.12236)).

**Early-stop and select on patient-level validation log loss (unweighted, after prior correction) or AUROC.** Sensitivity at FPR and AUPRC swing by several points per epoch with around 20 validation positives, so they suit reporting, not selection. Average fold models' logits and recalibrate the ensemble once, rather than averaging individually calibrated probabilities.

## Grouped nested CV and Neyman–Pearson thresholds keep estimates honest

**Split by patient and stratify on the patient outcome.** Use `StratifiedGroupKFold` with groups = labour or patient ID. It is greedy, so check per-fold positive-patient counts yourself. Before scikit-learn 1.8, `shuffle=True` silently degraded it to an unstratified GroupKFold ([issue #32478](https://github.com/scikit-learn/scikit-learn/issues/32478); [1.8 changelog](https://scikit-learn.org/stable/whats_new/v1.8.html)).

Everything learned from data is fit inside each outer training fold, on a grouped inner validation split or inner CV:

- normalisation statistics
- sampling weights
- early stopping
- hyperparameters
- the calibration map
- the FPR threshold
- alarm-rule parameters

The outer fold is touched once. This matters because tuning on the reporting CV is optimistic ([Varma & Simon 2006](https://doi.org/10.1186/1471-2105-7-91)), and selection bias can be as large as the gaps between algorithms ([Cawley & Talbot 2010](https://jmlr.org/papers/v11/cawley10a.html)). Oversampling before splitting leaks duplicated positives. So do segment-level random splits, even without oversampling ([Kapoor & Narayanan 2023](https://doi.org/10.1016/j.patter.2023.100804)).

A fixed inner validation fold is acceptable for deep models. The cost is fewer negatives for the threshold. Flat CV is often fine for *choosing* between models ([Wainer & Cawley 2021](https://doi.org/10.1016/j.eswa.2021.115222)), but the final number needs nested CV or a held-out cohort. Repeated grouped CV (e.g. 5×5) reduces partition variance when adverse patients are few.

**Pool the right quantities across folds.** Metrics computed across CV folds can be aggregated in several incompatible ways, and all but one are biased under imbalance ([Forman & Scholz 2010](https://dl.acm.org/doi/10.1145/1882471.1882479)). The recommendations:

- **AUROC:** report the per-fold mean ± SD as primary. Pooled out-of-fold AUROC is valid only after per-fold calibration puts scores on one scale.
- **Threshold metrics:** pool the confusion counts, since each patient is classified by its own fold's frozen rule.
- **Uncertainty:** CV estimates the error of the *procedure*, not of the final model. Naive CV intervals miscover at 2–3× the nominal rate. The nested-CV standard error restores coverage ([Bates, Hastie & Tibshirani 2024](https://arxiv.org/abs/2104.00673)).

**Pick the FPR threshold on validation negatives, count patients, and know how much the cap will leak.** The naive "smallest threshold with empirical FPR ≤ α" rule yields classifiers of which "only approximately half" meet α on the population. A CV-based estimate has the same problem. The Neyman–Pearson umbrella algorithm fixes this ([Tong, Feng & Li 2018](https://www.science.org/doi/10.1126/sciadv.aao1659)). Sort the n held-out negative scores and take the order statistic T_(k*), where k* is the smallest k with Σ_{j=k}^{n} C(n,j)(1−α)^j α^{n−j} ≤ δ. Then P(FPR > α) ≤ δ. The cost, at a 30% cap with δ=0.05, is a much lower realised FPR (and so lower sensitivity) when validation negatives are few:

| Validation negatives n | Naive E[FPR] | Naive P(FPR>0.30) | NP-umbrella E[FPR] |
|---|---|---|---|
| 20 | 0.333 | 0.61 | 0.143 |
| 50 | 0.314 | 0.57 | 0.196 |
| 100 | 0.307 | 0.55 | 0.228 |
| 200 | 0.303 | 0.53 | 0.244 |
| 500 | 0.301 | 0.52 | 0.265 |

These values come from the order-statistic Beta(n−k+1, k) distribution, computed in the research notes. Test-set binomial noise comes on top.

Two practical points follow:

- **Correlated negatives.** Segments from one labour are not i.i.d., so n must be the number of negative *patients*. Define the cap on a patient-level alarm score (the fraction of normal labours ever alarmed), or the guarantee is void.
- **Ties.** Saturated sigmoids create ties. Threshold on logits with a strict ">" rule and report the fraction tied.

MAPIE's `BinaryClassificationController(risk="fpr")` offers the same guarantee via Learn-Then-Test ([MAPIE](https://github.com/scikit-learn-contrib/MAPIE)), but the NP rule is about ten lines with `scipy.stats.binom`:

```python
k = next(k for k in range(1, n + 1) if binom.sf(k - 1, n, 1 - alpha) <= delta)
thr = np.sort(neg_scores)[k - 1]   # alarm if score > thr
```

Always report three separate quantities:

- (a) sensitivity and realised FPR at the validation-chosen threshold, which is the deployable number;
- (b) sensitivity read at exactly FPR=α on the test ROC, labelled as an oracle;
- (c) McClish-standardised pAUC over [0, α], via `roc_auc_score(max_fpr=α)`.

**Use a set of metrics that STRATOS would endorse.** The STRATOS review of 32 measures names as essential: AUROC, a calibration plot, net benefit with decision-curve analysis, and probability distributions per outcome class. It finds F1 fails both the properness and the decision-analytic criteria ([Van Calster et al. 2024](https://arxiv.org/abs/2412.10288)).

For the binary task, report:

- patient-level AUROC and pAUC;
- sensitivity, specificity, PPV and NPV at the pre-specified cap;
- the calibration set above;
- net benefit, NB = TP/N − (FP/N)·p_t/(1−p_t) ([Vickers & Elkin 2006](https://doi.org/10.1177/0272989X06295361));
- AUPRC, always drawn against its prevalence baseline ([Saito & Rehmsmeier 2015](https://doi.org/10.1371/journal.pone.0118432)).

AUPRC's superiority under imbalance is contested ([McDermott et al. 2024](https://arxiv.org/abs/2401.06091)), so keep AUROC for comparing models.

For the 3-class task, report:

- the 3×3 confusion matrix;
- macro one-vs-rest AUROC and Hand–Till AUROC ([Hand & Till 2001](https://doi.org/10.1023/A:1010920819831));
- quadratic-weighted kappa ([Cohen 1968](https://doi.org/10.1037/h0026256));
- ranked probability score;
- the two clinically meaningful binary collapses.

**Build CIs by resampling patients, with threshold selection inside the bootstrap.** Resampling clusters is consistent under within-cluster correlation ([Field & Welsh 2007](https://doi.org/10.1111/j.1467-9868.2007.00593.x)). For threshold-dependent metrics, each replicate should resample validation patients, re-derive the calibration and threshold, then resample test patients. The CI then carries threshold noise, which a fixed-threshold bootstrap ignores. This nested recipe is assembled from the cited pieces; no published procedure covers it directly.

Other interval rules:

- Wilson intervals for single proportions at patient level ([Brown, Cai & DasGupta 2001](https://doi.org/10.1214/ss/1009213286)).
- DeLong on one score per patient ([DeLong et al. 1988](https://doi.org/10.2307/2531595)), or Obuchowski's clustered AUC variance ([Obuchowski 1997](https://doi.org/10.2307/2533958)).
- The Nadeau–Bengio corrected resampled t-test for across-fold model comparison, whose variance term is (1/J + 1/(k−1))·σ̂² for k-fold CV ([Nadeau & Bengio 2003](https://doi.org/10.1023/A:1024068626366)).

**Evaluate continuous prediction the way it would run.** Last-window evaluation uses future information, because a clinician never knows which window is last. For patient-level early warning, borrow the ICU reporting conventions:

- event sensitivity: adverse labours alarmed before delivery, ideally ≥30 minutes before;
- a cumulative-detection-versus-lead-time curve;
- patient-level FPR;
- false alarms per monitored hour and per normal labour. Hyland reported 0.05 alarms per patient-hour ([Hyland et al. 2020](https://www.nature.com/articles/s41591-020-0789-4));
- alarms per true detection ([Tomašev et al. 2019](https://doi.org/10.1038/s41586-019-1390-1));
- time-resolved AUROC in bins of time before delivery (0–30, 30–60, 60–120, >120 min), with n per bin, stratified by stage, and with warm-up scores masked.

Candidate alarm rules are first crossing, k-of-n persistence, running max or EMA, and CUSUM-style accumulation. Their parameters are hyperparameters and stay in the inner loop. A PhysioNet-2019-style utility score, which rewards timely detection and penalises late or false alarms, can supplement these metrics but not replace them, because its weights are arbitrary ([PhysioNet 2019](https://moody-challenge.physionet.org/2019/)). No CTG-specific early-warning evaluation guideline exists. These are adaptations by analogy.

## TRIPOD+AI frames the write-up, and the existing stack covers nearly everything

Report against **TRIPOD+AI** ([Collins et al., BMJ 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/); [expanded checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf)) and self-audit against **PROBAST+AI**, which flags leakage, unaddressed optimism, sample size and calibration as sources of bias ([Moons et al., BMJ 2025](https://research.birmingham.ac.uk/en/publications/probastai-an-updated-quality-risk-of-bias-and-applicability-asses/)). CLAIM 2024 is imaging-specific and applies only loosely ([Tejani et al. 2024](https://pubs.rsna.org/doi/full/10.1148/ryai.240300)). DECIDE-AI applies only at a later live clinical evaluation ([Vasey et al. 2022](https://www.clinicalradiologyonline.net/article/S0009-9260(22)00704-8/fulltext)).

For this study the write-up must state:

- **Data flow and outcome.** Patient and recording flow with exclusions. The outcome definition and its timing relative to prediction, including exactly how the pH bands map to the 3 classes.
- **Inputs.** Segment length, stride and signal-loss handling, with all preprocessing fit on training folds.
- **Sample size.** Events per class, justified per the Riley–Collins sample-size guidance ([Riley et al. 2024](https://pubmed.ncbi.nlm.nih.gov/38253388/)).
- **Pre-specified analysis.** The search budget, the grouped CV scheme, the calibration method, the FPR cap and NP δ, and the alarm rule.
- **Framing.** Prediction and observation windows and trigger logic. Lauritsen proposes extending TRIPOD item 7a to cover exactly this ([Lauritsen 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8593052/)).
- **Subgroups.** Performance by parity, gestational age, stage and delivery mode.
- **Limitations.** An explicit statement that internal k-fold validation must be followed by temporal or external validation. CTG's 0.81 vs 0.65 cross-site asymmetry makes that non-negotiable.

On tooling, the project venv (Python 3.14, scikit-learn 1.9, torch 2.14, scipy 1.18, Lightning 2.6.5) already contains almost everything. It does **not** match `requirements.txt`, which pins sklearn 1.8, torch 2.7.1 and pandas 2.3. Reconcile the two first, because several of the helpers below need sklearn ≥1.8 or 1.9, and pandas 3 changes the default string dtype of patient-ID columns ([PyPI scikit-learn](https://pypi.org/project/scikit-learn/); [pandas 3.0](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html)).

| Need | Use | Avoid / note |
|---|---|---|
| Grouped stratified folds | `StratifiedGroupKFold` (≥1.8 for correct `shuffle`) | Check per-fold patient counts manually |
| pAUC, operating points | `roc_auc_score(max_fpr=)`, `confusion_matrix_at_thresholds` (1.8), `metric_at_thresholds` (1.9) ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.metric_at_thresholds.html)) | `TunedThresholdClassifierCV` is awkward around Lightning; `roc_curve` drops collinear thresholds unless `drop_intermediate=False` |
| Calibration | `CalibratedClassifierCV(FrozenEstimator(...), method='sigmoid' or 'temperature')`, or a direct Platt/LBFGS fit on out-of-fold logits ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)) | `cv='prefit'` is removed in 1.9; netcal drags in pyro, gpytorch and tensorboard |
| Calibration plots | `calibration_curve(strategy='quantile')`; optionally `relplot` smooth ECE ([apple/ml-calibration](https://github.com/apple/ml-calibration)) | Equal-width bins at low prevalence |
| In-training monitoring | torchmetrics `BinaryAUROC`, matches sklearn exactly | Auto-sigmoid silently treats in-range logits as probabilities; `BootStrapper` ignores patient grouping; DDP test replicates samples ([torchmetrics](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html)) |
| Imbalance | Unweighted BCE; `pos_weight`; `torchvision.ops.sigmoid_focal_loss` (override α); `WeightedRandomSampler` | imbalanced-learn (SMOTE on latents is meaningless and leaks across patients) |
| Ordinal head | Copy CORAL/CORN from coral-pytorch (tiny, dependency-free, last release 2022) ([GitHub](https://github.com/raschka-research-group/coral_pytorch)) | — |
| AUC/pAUC losses | Copy one LibAUC loss class if ablating ([LibAUC](https://github.com/Optimization-AI/LibAUC)) | Installing LibAUC pulls torch_geometric, transformers, opencv |
| MIL pooling, masking | Hand-written gated attention (~25 lines, based on torchmil's `AttentionPool`) plus `nn.TransformerEncoder(src_key_padding_mask=)` ([torchmil](https://github.com/Franblueee/torchmil)) | tsai, PyHealth, aeon: version conflicts or TensorFlow-based |
| FPR guarantee | NP umbrella via `scipy.stats.binom`; optionally MAPIE 1.5 `BinaryClassificationController` ([PyPI](https://pypi.org/project/MAPIE/)) | Python `nproc` (stale, wraps its own training) |
| CIs and tests | `scipy.stats.bootstrap` on per-patient statistics; `binomtest(...).proportion_ci('wilson')`; copied fast DeLong ([roc_comparison](https://github.com/yandexdataschool/roc_comparison), check licence) | dcurves (no Python 3.14 support); net benefit is about 10 lines |
| Experiment loop | Outer Python fold loop, nested MLflow runs, `seed_everything(workers=True)`, `Trainer(deterministic=True)` ([Lightning k-fold example](https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/fabric/kfold_cv)) | LightningCLI/Hydra; `mlflow.pytorch.autolog` is only tested to torch 2.13 ([mlflow](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html)) |

The single most useful engineering decision is one long-format prediction table, with one row per scored segment. Its columns: run, seed, fold, split, patient and recording IDs, segment index, minutes since onset and before delivery, stage, label and raw pH, logit, probability and calibrated probability. Every downstream analysis reads only this table: thresholds, the nested bootstrap, DeLong, calibration, decision curves and time-resolved AUROC. That separates training from evaluation and makes the TRIPOD+AI reporting auditable.

## Conclusion

The field's open problems sit exactly where this project's design choices fall. The following are all unpublished for CTG, so each is a contribution rather than a replication:

- **Attention-MIL over segment latents** addresses FHR-LINet's stated equal-weighting limitation.
- **Per-token conditioning on time since onset and stage** replaces the literature's separate stage models.
- **A cumulative-link ordinal head** turns the three pH bands into one ranking and makes the DeepCTG-style dual binary tasks a free by-product.
- **Horizon-weighted segment labels** give weak supervision a principled form.

The same novelty means claims must rest on evaluation that reviewers cannot fault. The evidence here says evaluation, more than modelling, is where CTG results are won or lost.

Two tensions deserve deliberate decisions, not defaults. First, a 30% FPR cap doubles clinical practice. With a few hundred validation patients, the naive threshold will overshoot it about half the time, while the NP-guaranteed threshold will sit near 20–25% realised FPR. Pick one in advance and report both views. Second, a VAE whose KL and reconstruction signals are meant to be interpretable should not be co-trained with the classifier by default. The tuning burden is high, the classification benefit of semi-supervised VAEs mostly comes from the discriminative path anyway, and a frozen or stop-gradient encoder keeps the generative story intact. Supervised gains, if needed, should come through LP-FT on the top block. Finally, internal patient-grouped CV on one cohort measures the procedure, not the model. The 0.16-AUROC cross-site gap in CTG says an external or temporal cohort is the next necessary step, not an optional one.
