Yes — after pretraining, the downstream problem should **not** be treated as a single static classification on a completed signal.
It should be treated as a **causal sequential risk estimation problem**:

> at each time (t), given everything observed up to (t), estimate the probability that this baby is unhealthy, and estimate the likely severity, while updating that estimate as more data arrives.

That is the right framing if the clinical goal is **early detection**, not retrospective classification.

So the downstream model should be an **online classifier over causal pretrained embeddings**, not a one-shot classifier over a full-window pooled embedding.

---

# 1. Core downstream strategy

The correct downstream strategy is:

## **pretrained causal encoder (\rightarrow) sequence of causal embeddings (\rightarrow) online recurrent classifier (\rightarrow) risk at every decision time**

This is the key change from the earlier “full-window embedding” view.

During pretraining, we were learning a causal representation.
For downstream online detection, we must **use that representation causally**, not collapse the entire window with future-aware pooling.

So the downstream system should operate as follows:

1. process the signal causally,
2. at each decision time extract a causal embedding,
3. update a small online classifier state,
4. output:

   * binary unhealthy risk,
   * and severity risk,
5. repeat as time advances.

This lets you classify **as you go forward in time**.

---

# 2. What should be the input of the classifier?

The classifier should **not** take the raw scattering sequences again if the encoder is already pretrained well.
The classifier should take the **causal latent summaries produced by the pretrained model**.

Under the final first-iteration pretraining design, the natural per-time-step embedding is:

$$
x_t
===

\big[
s_t^{F}
;|;
s_t^{FU}
;|;
\mu_t^{TE}
\big].
$$

where:

* (s_t^{F} \in \mathbb{R}^{d}): causal summary of the recent (FHR)-only state,
* (s_t^{FU} \in \mathbb{R}^{d}): causal summary of the recent fused (FHR+UP) state,
* (\mu_t^{TE} \in \mathbb{R}^{d_z}): posterior mean of the local TE-style latent.

So if

$$
d = 192, \qquad d_z = 16,
$$

then

$$
x_t \in \mathbb{R}^{400}.
$$

This is the main downstream input.

---

## 2.1 Why this input is the right one

This input contains exactly the pieces you want the classifier to see:

### (s_t^{F})

captures intrinsic fetal dynamics:

* baseline regime,
* variability regime,
* autonomous abnormalities,
* deceleration/recovery behavior.

### (s_t^{FU})

captures multimodal predictive state:

* how the current uterine context and fetal context combine,
* stress-response structure,
* whether the current state is worsening.

### (\mu_t^{TE})

captures source-incremental influence:

* how much (UP) adds beyond (FHR),
* whether the uterine history is actually predictive of upcoming fetal changes.

This is much better than feeding only one of them.

---

## 2.2 Should we include (s_t^U) as well?

Usually, I would **not** make (s_t^U) part of the primary classifier input in the first version.

Why?

Because the job of (UP) is not to be classified for its own sake.
Its job is to explain how the fetus is responding.

That information should already be reflected in:

* (s_t^{FU}),
* (\mu_t^{TE}).

Feeding (s_t^U) directly can encourage the classifier to rely too much on uterine activity itself rather than on **uterine-conditioned fetal state**.

So for the first version, I would use

$$
x_t = [s_t^{F} | s_t^{FU} | \mu_t^{TE}]
$$

as the main input.

---

## 2.3 Should we include temporal deltas of the embedding?

Yes, this is useful.

Clinical deterioration is not only about current state, but about **change of state**.

So an even better classifier input is

$$
\tilde x_t
==========

\big[
x_t
;|;
(x_t - x_{t-1})
\big].
$$

Then if (x_t \in \mathbb{R}^{400}), you get

$$
\tilde x_t \in \mathbb{R}^{800}.
$$

This helps the classifier see:

* worsening trend,
* abrupt transitions,
* increasing coupling burden,
* progressive loss of variability.

I would include this delta term in the first downstream classifier.

---

# 3. What should be the decision times?

This is important.

Because you want **continuous classification**, you need a sequence of decision points.

The exact choice depends on what your data pipeline allows.

---

## 3.1 Practical first choice: classify once per new window

Since your current pipeline already creates overlapping windows with a 2-minute stride, the simplest downstream online sequence is:

* each new window = one decision point,
* decision every 2 minutes,
* each embedding summarizes the recent history ending at that window’s endpoint.

So for infant (i), you get a sequence

$$
x_{i,1}, x_{i,2}, \dots, x_{i,N_i},
$$

ordered in time toward delivery.

This is the most practical first downstream setup because it matches your existing data structure exactly.

---

## 3.2 Earliest possible detection under current pipeline

If your representation is based on a (20)-minute effective context, then with the current setup the first strong decision can only happen after the model has enough context.

So the earliest detection time is limited by the shortest context the encoder can use reliably.

If you only use the fixed (20)-minute windows exactly as currently defined, then:

* your first downstream decision occurs after the first valid window,
* and then it updates every 2 minutes.

That is already online and clinically more useful than a single terminal decision.

---

## 3.3 If you want earlier-than-window-end detection

Then later you should make the encoder explicitly operate on variable-length prefixes and train the downstream classifier on shorter prefixes too.

But for the **first iteration**, I recommend:

## **online classification at the level of successive overlapping windows**

That is the cleanest first downstream system.

---

# 4. What classifier should we use?

The classifier should be **causal, lightweight, stateful, and stable**.

My recommendation is:

## **a GRU-based online sequence classifier**

not an MLP, not a full transformer, not a static classifier.

---

## 4.1 Why not just an MLP on each window embedding?

A per-window MLP would compute

$$
p_t = \sigma(f(x_t))
$$

independently for each window.

That is too weak because it ignores the temporal evolution of evidence.

In this problem, the classifier needs to detect:

* repeated abnormal responses,
* worsening over time,
* sustained variability suppression,
* accumulation of bad signs.

A single window can be noisy.
What matters clinically is often the **trajectory**.

So independent window classification is not the right primary model.

It is fine as a baseline, but not as the main system.

---

## 4.2 Why not a transformer over window embeddings?

A transformer over window embeddings can work, but it is not the best first choice for online deployment.

You want something that:

* updates incrementally,
* is easy to train on limited medical data,
* naturally maintains a persistent state,
* is simple to interpret and debug.

A GRU does exactly that.

So for the first downstream classifier, use:

## **1- or 2-layer GRU**

with hidden size around

$$
d_h = 128 \text{ or } 256.
$$

---

# 5. Online downstream model

For infant (i), let the sequence of downstream inputs be

$$
\tilde x_{i,1}, \tilde x_{i,2}, \dots, \tilde x_{i,N_i}.
$$

The GRU evolves as

$$
h_{i,t}
=======

\operatorname{GRUCell}(\tilde x_{i,t}, h_{i,t-1}),
\qquad
h_{i,t} \in \mathbb{R}^{d_h}.
$$

This hidden state is the current online clinical state of the patient according to the model.

It accumulates:

* recent evidence,
* repeated patterns,
* deterioration trajectory,
* persistence of abnormality.

That is exactly what you want.

---

# 6. Outputs of the classifier

You want at least two outputs:

1. **binary unhealthy risk**
2. **severity estimate**

---

## 6.1 Binary unhealthy risk

Define a binary logit

$$
b_{i,t} = w_b^\top h_{i,t} + c_b,
$$

and probability

$$
p_{i,t}^{bin}
=============

\sigma(b_{i,t}).
$$

Interpretation:

$$
p_{i,t}^{bin}
\approx
P(\text{unhealthy baby} \mid \text{data up to time } t).
$$

This is the main online risk signal.

---

## 6.2 Severity output

Severity is ordered, so it should not be modeled as an unordered softmax if you can avoid it.

Instead, use an **ordinal head**.

Let the severity classes be

$$
y^{sev} \in {0,1,\dots,K-1}
$$

for example:

* (0) = healthy,
* (1) = mildly unhealthy,
* (2) = more severe,
* etc.

Define a scalar severity score

$$
u_{i,t} = w_s^\top h_{i,t} + c_s.
$$

Use ordered thresholds

$$
\tau_1 < \tau_2 < \cdots < \tau_{K-1}.
$$

Then define cumulative ordinal probabilities

$$
q_{i,t}^{(k)}
=============

# P(y^{sev} \ge k \mid \text{data up to } t)

\sigma(u_{i,t} - \tau_k),
\qquad
k=1,\dots,K-1.
$$

This is much better than a flat multiclass softmax because:

* severity has order,
* the model should know that “severe” is farther from “healthy” than “mild” is,
* the risk trajectory should evolve consistently with severity.

So I recommend:

* binary head for unhealthy detection,
* ordinal head for severity.

---

# 7. How should we train this classifier?

This is the hardest part conceptually.

Why? Because you want **continuous online predictions**, but you likely only have **final infant-level labels**.

So the downstream supervision is weak in time.

The model must learn:

* how to output risk at every time step,
* without being forced to make unrealistically strong early predictions before enough evidence exists.

So you should **not** simply copy the final label to every prefix with equal weight.

That would make early prefixes noisy labels and encourage spurious early alarms.

---

# 8. Correct training strategy for online classification

The right training setup is:

## **prefix-supervised sequence training with time-dependent weighting**

For infant (i), with sequence length (N_i), the GRU outputs a risk at every time step:

$$
p_{i,1}^{bin}, \dots, p_{i,N_i}^{bin},
$$

and severity outputs

$$
q_{i,1}^{(k)}, \dots, q_{i,N_i}^{(k)}.
$$

You train these outputs against the infant label, but with **weights that depend on time**.

---

## 8.1 Why time-dependent weighting is necessary

If you use the final label equally at all times, then for an unhealthy infant the model is asked to say “unhealthy” even at very early times when there may be little evidence.

That creates label noise.

If you supervise only the final time point, the model may become accurate only very late.

So you need a compromise:

* strong supervision late,
* weaker supervision early,
* and a training schedule that gradually encourages earlier detection.

That is the correct logic.

---

# 9. Binary classification loss

Let the infant-level binary label be

$$
y_i^{bin} \in {0,1}.
$$

Define time weights

$$
\omega_{i,t} \ge 0,
\qquad
\sum_{t=1}^{N_i} \omega_{i,t} = 1.
$$

Use a weighted binary cross-entropy:

$$
\mathcal L_{bin}
================

\sum_i
\sum_{t=1}^{N_i}
\omega_{i,t}
\operatorname{BCE}(p_{i,t}^{bin}, y_i^{bin}).
$$

where

$$
\operatorname{BCE}(p,y)
=======================

-y\log p - (1-y)\log(1-p).
$$

Because unhealthy cases are often imbalanced, use class weights:

$$
\mathcal L_{bin}
================

\sum_i
\sum_{t=1}^{N_i}
\omega_{i,t}
\Big[
\alpha_1 y_i^{bin},(-\log p_{i,t}^{bin})
+
\alpha_0 (1-y_i^{bin}),(-\log(1-p_{i,t}^{bin}))
\Big].
$$

This is the main binary objective.

---

# 10. Ordinal severity loss

Let the severity label be

$$
y_i^{sev} \in {0,\dots,K-1}.
$$

For each threshold (k), define the cumulative target

$$
z_i^{(k)} = \mathbf{1}{y_i^{sev} \ge k},
\qquad k=1,\dots,K-1.
$$

Then the weighted ordinal loss is

$$
\mathcal L_{ord}
================

\sum_i
\sum_{t=1}^{N_i}
\omega_{i,t}
\sum_{k=1}^{K-1}
\operatorname{BCE}(q_{i,t}^{(k)}, z_i^{(k)}).
$$

This gives you severity prediction at every time step.

---

# 11. Temporal stability and monotonic evidence

For online detection, raw per-step risk outputs can be noisy.

You do not want a risk trajectory that wildly oscillates up and down from one decision time to the next, especially because your windows overlap strongly.

So the downstream objective should include stability constraints.

---

## 11.1 Smoothness loss

A simple smoothness penalty on the risk logits is

$$
\mathcal L_{smooth}
===================

\sum_i
\sum_{t=2}^{N_i}
\big(
b_{i,t} - b_{i,t-1}
\big)^2.
$$

This discourages unnecessary jitter.

Use this with a small weight only.
You do not want to suppress true rapid deterioration; you only want to reduce unstable oscillation.

---

## 11.2 Monotonicity bias for positive risk

For unhealthy infants, once evidence accumulates, you generally want the risk trajectory to move upward, not repeatedly fall back down.

You should not enforce a hard monotonic constraint, because clinical evidence can become temporarily ambiguous. But a **soft monotonic bias** is useful.

For positive cases only, define

$$
\mathcal L_{mono}
=================

\sum_{i : y_i^{bin}=1}
\sum_{t=2}^{N_i}
\max!\big(0,, p_{i,t-1}^{bin} - p_{i,t}^{bin}\big)^2.
$$

This penalizes decreases in risk for positive patients.

Use this weakly.

This helps the model behave like an early warning system rather than a fluctuating regressor.

---

# 12. Final downstream objective

The downstream fine-tuning objective should be

$$
\mathcal L_{down}
=================

\lambda_{bin}\mathcal L_{bin}
+
\lambda_{ord}\mathcal L_{ord}
+
\lambda_{smooth}\mathcal L_{smooth}
+
\lambda_{mono}\mathcal L_{mono}.
$$

A good first setting is:

$$
\lambda_{bin}=1,\qquad
\lambda_{ord}=0.5,\qquad
\lambda_{smooth}=10^{-3},\qquad
\lambda_{mono}=10^{-2}.
$$

The binary head should dominate. Severity is important, but detecting unhealthy babies early is the primary task.

---

# 13. How should the time weights (\omega_{i,t}) be chosen?

This is crucial.

You want:

* strong supervision late, because label reliability is highest late,
* but you also want the model to learn earlier detection.

So use a **curriculum**.

---

## 13.1 Late-heavy initial weighting

At the beginning of downstream training, define

$$
\omega_{i,t}^{(0)}
\propto
\left(\frac{t}{N_i}\right)^\gamma,
\qquad
\gamma > 1.
$$

For example:

$$
\gamma = 3.
$$

This means late steps get much higher weight.

That lets the classifier first learn clear late-stage patterns.

---

## 13.2 Gradually flatten the weights

As training progresses, reduce (\gamma) toward (1) or even lower, making supervision less late-concentrated.

So the model first learns “easy late detection,” then gradually learns to move that detection earlier.

This is the right training schedule for early warning under weak temporal labels.

---

# 14. Freeze or fine-tune the pretrained encoder?

Use a two-stage downstream strategy.

---

## 14.1 Stage 1: frozen encoder

First, freeze the pretrained encoder and train only the downstream GRU classifier.

Why?

Because this tells you whether the pretrained representation already contains clinically useful online information.

If performance is already good, you know the pretraining worked.

---

## 14.2 Stage 2: light fine-tuning

Then unfreeze:

* the top part of the causal encoders,
* and possibly the TE posterior/prior heads if needed,

and fine-tune end-to-end with a smaller learning rate.

This lets the representation adapt to the classification task without destroying the pretrained structure.

So the downstream procedure should be:

1. frozen encoder + GRU,
2. then partial fine-tuning.

---

# 15. How do we make decisions online?

At inference time, the model produces a risk at each step:

$$
p_t^{bin}, \qquad q_t^{(k)}.
$$

But you do not want to fire an alarm on every single noisy fluctuation.

So use an alarm policy.

---

## 15.1 Binary unhealthy alarm

Use a threshold (\tau_{on}).
Raise alarm if

$$
p_t^{bin} > \tau_{on}
$$

for (m) consecutive steps.

For example:

* (\tau_{on}=0.7),
* (m=2) or (3).

This reduces false alarms from transient spikes.

---

## 15.2 Hysteresis

To avoid alarm chattering, use a lower switch-off threshold (\tau_{off}):

* turn on if (p_t^{bin} > \tau_{on}),
* do not turn off unless (p_t^{bin} < \tau_{off}),

with

$$
\tau_{off} < \tau_{on}.
$$

This is important in online systems.

---

## 15.3 Severity output

At each step, the ordinal head gives severity probabilities.

You can either:

* report the most likely severity class,
* or report the cumulative probabilities (P(y\ge k)).

Clinically, the cumulative form is often more useful because it says:

* low risk,
* elevated risk,
* high/severe risk.

---

# 16. What is the best input granularity for the classifier?

For the first version, I recommend:

## **one embedding per overlapping window**

ordered by time.

Why this is best initially:

* it matches your current preprocessing,
* it gives a stable update every 2 minutes,
* it reduces excessive redundancy,
* it is much easier to implement and validate.

So the classifier sequence is

$$
\tilde x_{i,1}, \tilde x_{i,2}, \dots, \tilde x_{i,N_i}
$$

where each step corresponds to one overlapping epoch/window.

Later, if you want even earlier detection, you can move to finer-grained anchors within the continuous signal. But for the first downstream system, window-level sequential classification is the right choice.

---

# 17. What should the classifier **not** use?

Do **not** use:

* the full-window future-pooled embedding from any non-causal pathway,
* the final outcome label copied equally to every time step without weighting,
* an independent MLP per window as the main classifier,
* a terminal-only classifier that only looks at the last window,
* aggressive temporal smoothing that suppresses genuine deterioration.

The classifier must remain:

* causal,
* sequential,
* evidence-accumulating,
* and capable of early escalation.

---

# 18. First recommended downstream architecture

Here is the exact first downstream system I would build.

## Encoder output at decision time (t)

From the pretrained model extract:

$$
s_t^F \in \mathbb{R}^{192},\qquad
s_t^{FU} \in \mathbb{R}^{192},\qquad
\mu_t^{TE} \in \mathbb{R}^{16}.
$$

Form

$$
x_t = [s_t^F | s_t^{FU} | \mu_t^{TE}] \in \mathbb{R}^{400}.
$$

Then include the delta:

$$
\tilde x_t = [x_t | (x_t - x_{t-1})] \in \mathbb{R}^{800}.
$$

For (t=1), use zero delta.

## Online classifier state

Use a GRU:

$$
h_t = \operatorname{GRUCell}(\tilde x_t, h_{t-1}),
\qquad
h_t \in \mathbb{R}^{128} \text{ or } \mathbb{R}^{256}.
$$

## Binary head

$$
p_t^{bin} = \sigma(w_b^\top h_t + c_b).
$$

## Ordinal severity head

$$
u_t = w_s^\top h_t + c_s,
$$

$$
q_t^{(k)} = \sigma(u_t - \tau_k), \qquad k=1,\dots,K-1.
$$

This is the classifier I would use first.

---

# 19. Recommended training recipe

Use the following sequence:

### Step 1

Pretrain the causal encoder as already defined.

### Step 2

Freeze the encoder and extract causal embeddings for every window.

### Step 3

Train the GRU sequence classifier on per-infant ordered window sequences.

### Step 4

Use weighted prefix supervision with late-heavy weights first.

### Step 5

Gradually flatten the prefix weights so the classifier learns to move correct detections earlier.

### Step 6

Unfreeze the upper encoder layers and fine-tune end-to-end with a smaller learning rate.

This is the right strategy because it cleanly separates:

* representation learning,
* online decision learning,
* and task adaptation.

---

# 20. Final answer

After pretraining, you should **not** do a one-shot classification on the complete signal. The correct downstream formulation is **causal sequential risk prediction** over time. At each decision point (for the first version: each new overlapping window), extract a causal embedding
$$
x_t = [s_t^F | s_t^{FU} | \mu_t^{TE}],
$$
optionally augmented with its temporal difference, and feed the resulting sequence into a **small GRU-based online classifier**. This classifier should output at every step both a binary unhealthy probability
$$
p_t^{bin} = P(\text{unhealthy} \mid \text{data up to } t)
$$
and an ordered severity estimate through an ordinal head. Train it with **prefix-level weighted supervision**, not terminal-only supervision and not equal supervision at all times: early in training, give more weight to later prefixes, then gradually flatten the weighting so the model learns earlier detection without being dominated by noisy early labels. Add small temporal smoothness and monotonic-risk penalties so the online alarm is stable. In short: the classifier input should be the **causal pretrained embedding sequence**, the classifier should be a **causal recurrent model (GRU)**, and the output should be a **running risk trajectory** that can trigger early unhealthy detection as soon as sufficient evidence appears.


So based on above what should be the architecture of the classifier? We have time from labor onset and the delta t of gaps of 20 minute segments. What is in the input (give me full details and shape of the tensors at each step.)




## PART 2: 


Yes. Given everything above, the **right first downstream classifier** is a **time-aware causal recurrent classifier** that operates on **one pretrained embedding per 20-minute segment**, in chronological order from labor onset.

The key idea is:

* the pretrained encoder converts each 20-minute segment into a compact causal representation,
* the downstream classifier receives the sequence of these segment embeddings,
* it also receives **time from labor onset** and **the actual gap (\Delta t)** between consecutive segments,
* and it outputs an updated unhealthy risk and severity estimate **after every new segment**.

So this is **not** a one-shot classifier over a completed record. It is an **online segment-level sequence model**.

---

# 1. What is the basic unit seen by the classifier?

The classifier should not directly consume raw (FHR)-ST and (UP)-ST again as its main input.

Instead, for each segment (j), it should consume a **segment-level causal embedding** extracted by the pretrained model.

Let infant (i) have (N_i) available segments, indexed in temporal order:

$$
\big(Y_{i,1}, U_{i,1}\big),;
\big(Y_{i,2}, U_{i,2}\big),;
\dots,;
\big(Y_{i,N_i}, U_{i,N_i}\big),
$$

where each segment is one (20)-minute effective window represented by

$$
Y_{i,j} \in \mathbb{R}^{300 \times d_F},
\qquad
U_{i,j} \in \mathbb{R}^{300 \times d_U}.
$$

For each segment (j), the pretrained encoder produces a compact embedding

$$
e_{i,j} \in \mathbb{R}^{D_e}.
$$

This is what the classifier consumes.

---

# 2. What exactly should the segment embedding be?

For the **first classifier iteration**, I recommend using a **compact segment embedding** rather than a very large pooled representation.

For each segment (j), run the pretrained causal encoder and extract:

## 2.1 Intrinsic (FHR) segment summary

From the (FHR)-only causal state sequence

$$
H_{F,i,j}^c \in \mathbb{R}^{300 \times d},
\qquad d = 192,
$$

compute an attention-pooled summary

$$
s_{i,j}^{F}
===========

\operatorname{AttnPool}!\big(H_{F,i,j}^c\big)
\in \mathbb{R}^{192}.
$$

## 2.2 Fused multimodal segment summary

From the fused causal state sequence

$$
H_{FU,i,j}^c \in \mathbb{R}^{300 \times d},
$$

compute

$$
s_{i,j}^{FU}
============

\operatorname{AttnPool}!\big(H_{FU,i,j}^c\big)
\in \mathbb{R}^{192}.
$$

## 2.3 TE summary for the segment

Inside the segment, evaluate the TE posterior on a fixed anchor grid

$$
\mathcal A^* = {a_1,\dots,a_M},
$$

for example every (15) feature steps over the valid range.
If (M=16), then for each anchor (a_m) you have posterior mean

$$
\mu_{i,j,a_m}^{TE} \in \mathbb{R}^{16}.
$$

Then average these anchor-level TE means:

$$
\bar \mu_{i,j}^{TE}
===================

\frac{1}{M}\sum_{m=1}^{M} \mu_{i,j,a_m}^{TE}
\in \mathbb{R}^{16}.
$$

## 2.4 Final segment embedding

Define

$$
e_{i,j}
=======

\big[
s_{i,j}^{F}
;|;
s_{i,j}^{FU}
;|;
\bar \mu_{i,j}^{TE}
\big]
\in \mathbb{R}^{400}.
$$

So the base classifier input per segment is:

$$
D_e = 192 + 192 + 16 = 400.
$$

This is the segment embedding I recommend for the first downstream classifier.

It is compact, causal, and it already contains:

* intrinsic fetal state,
* multimodal predictive state,
* TE-style uterine contribution.

---

# 3. What time information should be added?

You said you have:

* time from labor onset,
* and the (\Delta t) gaps between consecutive (20)-minute segments.

That should absolutely be used.

The classifier needs to know not only **what** the segment embedding is, but also **when** in labor it occurs and **how long** it has been since the previous segment.

---

## 3.1 Define the time variables

For infant (i) and segment (j):

* let (\tau_{i,j}) be the time from labor onset to the **end** of segment (j),
* let
  $$
  \Delta \tau_{i,j}
  =================

  \tau_{i,j} - \tau_{i,j-1}
  $$
  be the elapsed time since the previous available segment,
* define
  $$
  \Delta \tau_{i,1} = 0.
  $$

Because the nominal stride is (20) minutes, define also the deviation from nominal spacing:

$$
\delta_{i,j}
============

\Delta \tau_{i,j} - 20 \text{ minutes}.
$$

Also define a missing-gap indicator:

$$
m_{i,j}
=======

\mathbf 1{\Delta \tau_{i,j} > 20 + \epsilon}.
$$

This indicator is useful because if there are larger-than-expected gaps, the classifier should know that continuity is weaker.

---

## 3.2 Raw time feature vector

Convert times to hours for scale stability:

$$
\tau_{i,j}^{hr} = \frac{\tau_{i,j}}{60},
\qquad
\Delta \tau_{i,j}^{hr} = \frac{\Delta \tau_{i,j}}{60},
\qquad
\delta_{i,j}^{hr} = \frac{\delta_{i,j}}{60}.
$$

Then define the raw time feature vector

$$
r_{i,j}
=======

\Big[
\tau_{i,j}^{hr},
;
\log(1+\tau_{i,j}^{hr}),
;
\Delta \tau_{i,j}^{hr},
;
\log(1+\Delta \tau_{i,j}^{hr}),
;
\delta_{i,j}^{hr},
;
m_{i,j}
\Big]
\in \mathbb{R}^{6}.
$$

This is the best first time feature set because it gives the classifier:

* absolute progression through labor,
* local elapsed time,
* gap abnormality,
* and a missingness cue.

---

# 4. Should we also include segment-to-segment embedding change?

Yes — definitely.

The downstream decision is not just about the current segment embedding.
It is also about whether the state is changing.

So define the segment-embedding difference

$$
\Delta e_{i,j}
==============

e_{i,j} - e_{i,j-1},
\qquad
\Delta e_{i,1} = 0.
$$

This gives the classifier access to:

* worsening trends,
* abrupt changes,
* progressive loss of variability,
* escalation of coupling burden.

Since (e_{i,j} \in \mathbb{R}^{400}), we get

$$
\Delta e_{i,j} \in \mathbb{R}^{400}.
$$

---

# 5. Final classifier input per segment

Now combine:

1. current segment embedding (e_{i,j} \in \mathbb{R}^{400}),
2. segment difference (\Delta e_{i,j} \in \mathbb{R}^{400}),
3. raw time feature vector (r_{i,j} \in \mathbb{R}^{6}).

---

## 5.1 Time embedding MLP

Pass the time vector through a small MLP:

$$
t_{i,j}
=======

\operatorname{MLP}*{time}(r*{i,j})
\in \mathbb{R}^{32}.
$$

A good first architecture is:

* Linear (6 \to 32)
* GELU
* Linear (32 \to 32)

So the time embedding is (32)-dimensional.

---

## 5.2 Concatenated classifier token

Now define the per-segment classifier token:

$$
x_{i,j}^{cat}
=============

\big[
e_{i,j}
;|;
\Delta e_{i,j}
;|;
t_{i,j}
\big]
\in \mathbb{R}^{832}.
$$

because

$$
400 + 400 + 32 = 832.
$$

---

# 6. Batched tensor shapes before the classifier

Suppose a minibatch contains (B) infants, each padded to the same number of segments (N_{\max}).

Then the tensors are:

## 6.1 Segment embeddings

$$
E \in \mathbb{R}^{B \times N_{\max} \times 400}.
$$

## 6.2 Segment deltas

$$
\Delta E \in \mathbb{R}^{B \times N_{\max} \times 400}.
$$

## 6.3 Raw time features

$$
R \in \mathbb{R}^{B \times N_{\max} \times 6}.
$$

## 6.4 Time embeddings

$$
T \in \mathbb{R}^{B \times N_{\max} \times 32}.
$$

## 6.5 Concatenated classifier tokens

$$
X^{cat}
=======

[E ;|; \Delta E ;|; T]
\in \mathbb{R}^{B \times N_{\max} \times 832}.
$$

## 6.6 Sequence mask

For padding, use

$$
M \in {0,1}^{B \times N_{\max}},
$$

where (M_{i,j}=1) means segment (j) exists and (0) means padding.

---

# 7. What classifier architecture should we use?

The best first classifier is:

## **Time-aware GRU classifier**

not just a GRU, but a GRU with explicit elapsed-time conditioning.

Why?

Because if there is a larger-than-usual gap between segments, the old hidden state should not persist unchanged. The classifier should know that the evidence has become temporally stale.

That is exactly what a time-aware recurrent update gives you.

---

# 8. Detailed classifier architecture

---

## 8.1 Input projection block

First project the (832)-dimensional token into a manageable model dimension:

$$
X_{proj}
========

\operatorname{Proj}(X^{cat})
\in \mathbb{R}^{B \times N_{\max} \times 256}.
$$

Use:

* Linear (832 \to 256)
* LayerNorm
* GELU
* Dropout

So per segment:

$$
x_{i,j}
\in \mathbb{R}^{256}.
$$

This is the actual input to the recurrent classifier core.

---

## 8.2 Time-decay gate

Use the time embedding (t_{i,j} \in \mathbb{R}^{32}) to compute a hidden-state decay gate:

$$
\gamma_{i,j}
============

\exp\Big(
-\operatorname{softplus}(W_\gamma t_{i,j} + b_\gamma)
\Big)
\in (0,1)^{256}.
$$

So the shape is:

$$
\Gamma \in \mathbb{R}^{B \times N_{\max} \times 256}.
$$

Interpretation:

* if (\Delta \tau_{i,j}) is small and normal, then (\gamma_{i,j}) is closer to (1),
* if the gap is large, then (\gamma_{i,j}) becomes smaller,
* which causes the recurrent memory to decay more before the next update.

This is the correct way to use the gap information.

---

## 8.3 Recurrent state update

Let the recurrent hidden size be

$$
d_h = 256.
$$

For each infant (i), process the sequence causally.

Initialize

$$
h_{i,0} = 0 \in \mathbb{R}^{256}.
$$

At segment (j), first decay the previous state:

$$
\tilde h_{i,j-1}
================

\gamma_{i,j} \odot h_{i,j-1}.
$$

Then update with a GRU cell:

$$
h_{i,j}
=======

\operatorname{GRUCell}(x_{i,j}, \tilde h_{i,j-1}),
\qquad
h_{i,j} \in \mathbb{R}^{256}.
$$

Collect all hidden states into

$$
H \in \mathbb{R}^{B \times N_{\max} \times 256}.
$$

This hidden state is the online patient state used for classification.

---

## 8.4 Output feature for prediction at each segment

I recommend using both:

* the accumulated recurrent state,
* and the current projected token.

So define

$$
o_{i,j}
=======

[h_{i,j} ;|; x_{i,j}]
\in \mathbb{R}^{512}.
$$

This is useful because:

* (h_{i,j}) gives long-term accumulated context,
* (x_{i,j}) preserves the most recent current-segment evidence directly.

Collect into

$$
O \in \mathbb{R}^{B \times N_{\max} \times 512}.
$$

This is the final feature passed to the prediction heads.

---

# 9. Prediction heads

The classifier should produce output **at every segment step**.

---

## 9.1 Binary unhealthy head

Use a linear head

$$
\ell_{i,j}^{bin}
================

W_{bin} o_{i,j} + b_{bin},
\qquad
\ell_{i,j}^{bin} \in \mathbb{R}.
$$

Over the batch:

$$
L^{bin}
\in \mathbb{R}^{B \times N_{\max} \times 1}.
$$

Then the unhealthy probability is

$$
p_{i,j}^{bin}
=============

\sigma(\ell_{i,j}^{bin}).
$$

This is the online unhealthy risk estimate at segment (j).

---

## 9.2 Severity head

For severity, I recommend an ordinal head.

Use a scalar severity score:

$$
u_{i,j}
=======

W_{sev} o_{i,j} + b_{sev},
\qquad
u_{i,j} \in \mathbb{R}.
$$

Then define learned ordered thresholds

$$
\tau_1 < \tau_2 < \cdots < \tau_{K-1}.
$$

For each threshold (k), compute

$$
q_{i,j}^{(k)}
=============

# \sigma(u_{i,j} - \tau_k)

P(y_{i}^{sev} \ge k \mid \text{data up to segment } j).
$$

So the severity output tensor is

$$
Q^{sev}
\in \mathbb{R}^{B \times N_{\max} \times (K-1)}.
$$

This is better than an unordered softmax because severity is ordered.

---

# 10. Full tensor flow from raw segment to classifier output

Now I will write the full pipeline with shapes at each step.

---

## 10.1 Per segment, from pretrained encoder

For segment (j) of infant (i):

### Raw segment inputs

$$
Y_{i,j} \in \mathbb{R}^{300 \times d_F},
\qquad
U_{i,j} \in \mathbb{R}^{300 \times d_U}.
$$

### Pretrained encoder outputs

$$
H_{F,i,j}^c \in \mathbb{R}^{300 \times 192},
\qquad
H_{FU,i,j}^c \in \mathbb{R}^{300 \times 192}.
$$

### Segment summaries

$$
s_{i,j}^{F} = \operatorname{AttnPool}(H_{F,i,j}^c) \in \mathbb{R}^{192},
$$

$$
s_{i,j}^{FU} = \operatorname{AttnPool}(H_{FU,i,j}^c) \in \mathbb{R}^{192}.
$$

### TE anchor means

For (M) anchors:

$$
\mu_{i,j,a_m}^{TE} \in \mathbb{R}^{16},
\qquad m=1,\dots,M.
$$

Average them:

$$
\bar \mu_{i,j}^{TE}
===================

\frac{1}{M}\sum_{m=1}^M \mu_{i,j,a_m}^{TE}
\in \mathbb{R}^{16}.
$$

### Segment embedding

$$
e_{i,j}
=======

[s_{i,j}^{F} | s_{i,j}^{FU} | \bar \mu_{i,j}^{TE}]
\in \mathbb{R}^{400}.
$$

---

## 10.2 Sequence construction over an infant

For infant (i) with (N_i) segments:

### Segment embedding sequence

$$
E_i
===

[e_{i,1}, \dots, e_{i,N_i}]
\in \mathbb{R}^{N_i \times 400}.
$$

### Delta sequence

$$
\Delta e_{i,j}
==============

e_{i,j} - e_{i,j-1},
\qquad
\Delta e_{i,1}=0,
$$

so

$$
\Delta E_i \in \mathbb{R}^{N_i \times 400}.
$$

### Raw time-feature sequence

$$
R_i
===

[r_{i,1}, \dots, r_{i,N_i}]
\in \mathbb{R}^{N_i \times 6}.
$$

### Time embedding sequence

$$
T_i
===

\operatorname{MLP}_{time}(R_i)
\in \mathbb{R}^{N_i \times 32}.
$$

### Concatenated classifier sequence

$$
X_i^{cat}
=========

[E_i | \Delta E_i | T_i]
\in \mathbb{R}^{N_i \times 832}.
$$

### Projected classifier tokens

$$
X_i
===

\operatorname{Proj}(X_i^{cat})
\in \mathbb{R}^{N_i \times 256}.
$$

### Time-decay gates

$$
\Gamma_i
========

\operatorname{DecayNet}(T_i)
\in \mathbb{R}^{N_i \times 256}.
$$

### Recurrent hidden states

$$
H_i
===

[h_{i,1}, \dots, h_{i,N_i}]
\in \mathbb{R}^{N_i \times 256}.
$$

### Output features

$$
O_i
===

[o_{i,1}, \dots, o_{i,N_i}]
\in \mathbb{R}^{N_i \times 512}.
$$

### Binary logits

$$
L_i^{bin}
\in \mathbb{R}^{N_i \times 1}.
$$

### Binary probabilities

$$
P_i^{bin}
\in \mathbb{R}^{N_i \times 1}.
$$

### Severity outputs

$$
Q_i^{sev}
\in \mathbb{R}^{N_i \times (K-1)}.
$$

---

## 10.3 Batched version

After padding to (N_{\max}):

* segment embeddings:
  $$
  E \in \mathbb{R}^{B \times N_{\max} \times 400}
  $$
* deltas:
  $$
  \Delta E \in \mathbb{R}^{B \times N_{\max} \times 400}
  $$
* raw time features:
  $$
  R \in \mathbb{R}^{B \times N_{\max} \times 6}
  $$
* time embeddings:
  $$
  T \in \mathbb{R}^{B \times N_{\max} \times 32}
  $$
* concatenated tokens:
  $$
  X^{cat} \in \mathbb{R}^{B \times N_{\max} \times 832}
  $$
* projected tokens:
  $$
  X \in \mathbb{R}^{B \times N_{\max} \times 256}
  $$
* decay gates:
  $$
  \Gamma \in \mathbb{R}^{B \times N_{\max} \times 256}
  $$
* hidden states:
  $$
  H \in \mathbb{R}^{B \times N_{\max} \times 256}
  $$
* output features:
  $$
  O \in \mathbb{R}^{B \times N_{\max} \times 512}
  $$
* binary logits:
  $$
  L^{bin} \in \mathbb{R}^{B \times N_{\max} \times 1}
  $$
* binary probabilities:
  $$
  P^{bin} \in \mathbb{R}^{B \times N_{\max} \times 1}
  $$
* severity probabilities:
  $$
  Q^{sev} \in \mathbb{R}^{B \times N_{\max} \times (K-1)}
  $$

---

# 11. Why this classifier architecture is the right one

This architecture is the right first classifier because it uses exactly the information that matters for online detection:

* **current segment state** via (e_{i,j}),
* **change of state** via (\Delta e_{i,j}),
* **where in labor we are** via (\tau_{i,j}),
* **how much time elapsed since the last segment** via (\Delta \tau_{i,j}),
* **persistent accumulated evidence** via the recurrent hidden state (h_{i,j}),
* **irregular segment spacing** via the decay gate (\gamma_{i,j}).

This is much better than:

* an MLP on each segment independently,
* a static classifier on the full trajectory,
* or a recurrent model that ignores timing irregularity.

---

# 12. Online inference behavior

At test time, the model behaves as follows.

After each newly available segment (j):

1. compute its pretrained segment embedding (e_j),
2. compute (\Delta e_j),
3. compute time features (r_j),
4. compute projected token (x_j),
5. decay previous hidden state using (\Delta \tau_j),
6. update the GRU hidden state,
7. output:

   * current unhealthy risk (p_j^{bin}),
   * current severity estimate (q_j^{(k)}).

So the classifier can raise risk **as soon as evidence accumulates**, rather than waiting until the full signal is complete.

---

# 13. Final recommended classifier architecture

The classifier I recommend for the first downstream online system is:

## **Segment-level time-aware GRU classifier**

with the following exact structure:

### Per segment input

* pretrained segment embedding:
  $$
  e_j \in \mathbb{R}^{400}
  $$
* segment delta:
  $$
  \Delta e_j \in \mathbb{R}^{400}
  $$
* raw time feature vector:
  $$
  r_j \in \mathbb{R}^{6}
  $$

### Time MLP

$$
\mathbb{R}^{6} \to \mathbb{R}^{32}
$$

### Concatenated token

$$
x_j^{cat} \in \mathbb{R}^{832}
$$

### Input projection block

$$
\mathbb{R}^{832} \to \mathbb{R}^{256}
$$

### Decay gate

$$
\gamma_j \in \mathbb{R}^{256}
$$

### Recurrent core

GRU hidden state

$$
h_j \in \mathbb{R}^{256}
$$

### Output feature

$$
o_j = [h_j | x_j] \in \mathbb{R}^{512}
$$

### Heads

* binary unhealthy head:
  $$
  \mathbb{R}^{512} \to \mathbb{R}^{1}
  $$
* ordinal severity head:
  $$
  \mathbb{R}^{512} \to \mathbb{R}^{1}
  $$
  plus learned ordered thresholds.

That is the architecture I would implement first.

If you want, I can next write this as a **complete PyTorch-style module specification** with class definitions, `forward()` signatures, and exact code-level tensor shapes.

