 Time-aware causal recurrent classifier that operates on one pretrained embedding per 20-minute segment, in chronological order from labor onset.

The key idea is:
* the pretrained encoder converts each 20-minute segment into a compact causal representation,
* the downstream classifier receives the sequence of these segment embeddings,
* it also receives **time from labor onset** and **the actual gap $\Delta t$** between consecutive segments,
* and it outputs an updated unhealthy risk and severity estimate **after every new segment**.
---
# 1. What is the basic unit seen by the classifier?

Let infant $i$ have $N_i$ available segments, indexed in temporal order:

$$
\big(Y_{i,1}, U_{i,1}\big), \big(Y_{i,2}, U_{i,2}\big), \dots, \big(Y_{i,N_i}, U_{i,N_i}\big),
$$

where each segment is one $20$-minute effective window represented by

$$
Y_{i,j} \in \mathbb{R}^{300 \times d_F}, \qquad U_{i,j} \in \mathbb{R}^{300 \times d_U}.
$$

For each segment $j$, the pretrained encoder produces a compact embedding

$$
e_{i,j} \in \mathbb{R}^{D_e}.
$$

This is what the classifier consumes.

---

# 2. What exactly should the segment embedding be?

For the **first classifier iteration**, I recommend using a **compact segment embedding** rather than a very large pooled representation.

For each segment $j$, run the pretrained causal encoder and extract:

## 2.1 Intrinsic (FHR) segment summary

From the (FHR)-only causal state sequence

$$
H_{F,i,j}^c \in \mathbb{R}^{300 \times d}, \qquad d = 192,
$$

compute an attention-pooled summary

$$
s_{i,j}^{F} = \operatorname{AttnPool}\!\big(H_{F,i,j}^c\big) \in \mathbb{R}^{192}.
$$

## 2.2 Fused multimodal segment summary

From the fused causal state sequence

$$
H_{FU,i,j}^c \in \mathbb{R}^{300 \times d},
$$

compute

$$
s_{i,j}^{FU} = \operatorname{AttnPool}\!\big(H_{FU,i,j}^c\big) \in \mathbb{R}^{192}.
$$

## 2.3 TE summary for the segment

Inside the segment, evaluate the TE posterior on a fixed anchor grid

$$
\mathcal{A}^* = \{a_1,\dots,a_M\},
$$

for example every $15$ feature steps over the valid range.
If $M=16$, then for each anchor $a_m$ you have posterior mean

$$
\mu_{i,j,a_m}^{TE} \in \mathbb{R}^{16}.
$$

Then average these anchor-level TE means:

$$
\bar{\mu}_{i,j}^{TE} = \frac{1}{M}\sum_{m=1}^{M} \mu_{i,j,a_m}^{TE} \in \mathbb{R}^{16}.
$$

## 2.4 Final segment embedding

Define

$$
e_{i,j} = \big[ s_{i,j}^{F} \;|\; s_{i,j}^{FU} \;|\; \bar{\mu}_{i,j}^{TE} \big] \in \mathbb{R}^{400}.
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

* time from labor onset,
* and the $\Delta t$ gaps between consecutive $20$-minute segments.

The classifier needs to know not only **what** the segment embedding is, but also **when** in labor it occurs and **how long** it has been since the previous segment.

---
## 3.1 Define the time variables

For infant $i$ and segment $j$:

* let $\tau_{i,j}$ be the time from labor onset to the **end** of segment $j$,
* let
  $$
  \Delta \tau_{i,j} = \tau_{i,j} - \tau_{i,j-1}
  $$
  be the elapsed time since the previous available segment,
* define
  $$
  \Delta \tau_{i,1} = 0.
  $$

Because the nominal stride is $20$ minutes, define also the deviation from nominal spacing:

$$
\delta_{i,j} = \Delta \tau_{i,j} - 20 \text{ minutes}.
$$

Also define a missing-gap indicator:

$$
m_{i,j} = \mathbf{1}[\Delta \tau_{i,j} > 20 + \epsilon].
$$

This indicator is useful because if there are larger-than-expected gaps, the classifier should know that continuity is weaker.

---

## 3.2 Raw time feature vector

Convert times to hours for scale stability:

$$
\tau_{i,j}^{hr} = \frac{\tau_{i,j}}{60}, \qquad \Delta \tau_{i,j}^{hr} = \frac{\Delta \tau_{i,j}}{60}, \qquad \delta_{i,j}^{hr} = \frac{\delta_{i,j}}{60}.
$$

Then define the raw time feature vector

$$
r_{i,j} = \Big[ \tau_{i,j}^{hr},\; \log(1+\tau_{i,j}^{hr}),\; \Delta \tau_{i,j}^{hr},\; \log(1+\Delta \tau_{i,j}^{hr}),\; \delta_{i,j}^{hr},\; m_{i,j} \Big] \in \mathbb{R}^{6}.
$$

This is the best first time feature set because it gives the classifier:

* absolute progression through labor,
* local elapsed time,
* gap abnormality,
* and a missingness cue.

---

# 4. Should we also include segment-to-segment embedding change?

The downstream decision is not just about the current segment embedding.
It is also about whether the state is changing.

So define the segment-embedding difference

$$
\Delta e_{i,j} = e_{i,j} - e_{i,j-1}, \qquad \Delta e_{i,1} = 0.
$$

This gives the classifier access to:

* worsening trends,
* abrupt changes,
* progressive loss of variability,
* escalation of coupling burden.

Since $e_{i,j} \in \mathbb{R}^{400}$, we get

$$
\Delta e_{i,j} \in \mathbb{R}^{400}.
$$

---

# 5. Final classifier input per segment

Now combine:

1. current segment embedding $e_{i,j} \in \mathbb{R}^{400}$,
2. segment difference $\Delta e_{i,j} \in \mathbb{R}^{400}$,
3. raw time feature vector $r_{i,j} \in \mathbb{R}^{6}$.

---

## 5.1 Time embedding MLP

Pass the time vector through a small MLP:

$$
t_{i,j} = \operatorname{MLP}_{time}(r_{i,j}) \in \mathbb{R}^{32}.
$$

A good first architecture is:

* Linear $6 \to 32$
* GELU
* Linear $32 \to 32$

So the time embedding is $32$-dimensional.

---

## 5.2 Concatenated classifier token

Now define the per-segment classifier token:

$$
x_{i,j}^{cat} = \big[ e_{i,j} \;|\; \Delta e_{i,j} \;|\; t_{i,j} \big] \in \mathbb{R}^{832}.
$$

because

$$
400 + 400 + 32 = 832.
$$

---

# 6. Batched tensor shapes before the classifier

Suppose a minibatch contains $B$ infants, each padded to the same number of segments $N_{\max}$.

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
X^{cat} = [E \;|\; \Delta E \;|\; T] \in \mathbb{R}^{B \times N_{\max} \times 832}.
$$

## 6.6 Sequence mask

For padding, use

$$
M \in \{0,1\}^{B \times N_{\max}},
$$

where $M_{i,j}=1$ means segment $j$ exists and $0$ means padding.

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

First project the $832$-dimensional token into a manageable model dimension:

$$
X_{proj} = \operatorname{Proj}(X^{cat}) \in \mathbb{R}^{B \times N_{\max} \times 256}.
$$

Use:

* Linear $832 \to 256$
* LayerNorm
* GELU
* Dropout

So per segment:

$$
x_{i,j} \in \mathbb{R}^{256}.
$$

This is the actual input to the recurrent classifier core.

---

## 8.2 Time-decay gate

Use the time embedding $t_{i,j} \in \mathbb{R}^{32}$ to compute a hidden-state decay gate:

$$
\gamma_{i,j} = \exp\Big( -\operatorname{softplus}(W_\gamma t_{i,j} + b_\gamma) \Big) \in (0,1)^{256}.
$$

So the shape is:

$$
\Gamma \in \mathbb{R}^{B \times N_{\max} \times 256}.
$$

Interpretation:

* if $\Delta \tau_{i,j}$ is small and normal, then $\gamma_{i,j}$ is closer to $1$,
* if the gap is large, then $\gamma_{i,j}$ becomes smaller,
* which causes the recurrent memory to decay more before the next update.

This is the correct way to use the gap information.

---

## 8.3 Recurrent state update

Let the recurrent hidden size be

$$
d_h = 256.
$$

For each infant $i$, process the sequence causally.

Initialize

$$
h_{i,0} = 0 \in \mathbb{R}^{256}.
$$

At segment $j$, first decay the previous state:

$$
\tilde{h}_{i,j-1} = \gamma_{i,j} \odot h_{i,j-1}.
$$

Then update with a GRU cell:

$$
h_{i,j} = \operatorname{GRUCell}(x_{i,j}, \tilde{h}_{i,j-1}), \qquad h_{i,j} \in \mathbb{R}^{256}.
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
o_{i,j} = [h_{i,j} \;|\; x_{i,j}] \in \mathbb{R}^{512}.
$$

This is useful because:

* $h_{i,j}$ gives long-term accumulated context,
* $x_{i,j}$ preserves the most recent current-segment evidence directly.

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
\ell_{i,j}^{bin} = W_{bin} o_{i,j} + b_{bin}, \qquad \ell_{i,j}^{bin} \in \mathbb{R}.
$$

Over the batch:

$$
L^{bin} \in \mathbb{R}^{B \times N_{\max} \times 1}.
$$

Then the unhealthy probability is

$$
p_{i,j}^{bin} = \sigma(\ell_{i,j}^{bin}).
$$

This is the online unhealthy risk estimate at segment $j$.

---

## 9.2 Severity head
Do not implement in the first version


---

# 10. Full tensor flow from raw segment to classifier output

Now I will write the full pipeline with shapes at each step.

---

## 10.1 Per segment, from pretrained encoder

For segment $j$ of infant $i$:

### Raw segment inputs

$$
Y_{i,j} \in \mathbb{R}^{300 \times d_F}, \qquad U_{i,j} \in \mathbb{R}^{300 \times d_U}.
$$

### Pretrained encoder outputs

$$
H_{F,i,j}^c \in \mathbb{R}^{300 \times 192}, \qquad H_{FU,i,j}^c \in \mathbb{R}^{300 \times 192}.
$$

### Segment summaries

$$
s_{i,j}^{F} = \operatorname{AttnPool}(H_{F,i,j}^c) \in \mathbb{R}^{192},
$$

$$
s_{i,j}^{FU} = \operatorname{AttnPool}(H_{FU,i,j}^c) \in \mathbb{R}^{192}.
$$

### TE anchor means

For $M$ anchors:

$$
\mu_{i,j,a_m}^{TE} \in \mathbb{R}^{16}, \qquad m=1,\dots,M.
$$

Average them:

$$
\bar{\mu}_{i,j}^{TE} = \frac{1}{M}\sum_{m=1}^M \mu_{i,j,a_m}^{TE} \in \mathbb{R}^{16}.
$$

### Segment embedding

$$
e_{i,j} = [s_{i,j}^{F} \;|\; s_{i,j}^{FU} \;|\; \bar{\mu}_{i,j}^{TE}] \in \mathbb{R}^{400}.
$$

---

## 10.2 Sequence construction over an infant

For infant $i$ with $N_i$ segments:

### Segment embedding sequence

$$
E_i = [e_{i,1}, \dots, e_{i,N_i}] \in \mathbb{R}^{N_i \times 400}.
$$

### Delta sequence

$$
\Delta e_{i,j} = e_{i,j} - e_{i,j-1}, \qquad \Delta e_{i,1}=0,
$$

so

$$
\Delta E_i \in \mathbb{R}^{N_i \times 400}.
$$

### Raw time-feature sequence

$$
R_i = [r_{i,1}, \dots, r_{i,N_i}] \in \mathbb{R}^{N_i \times 6}.
$$

### Time embedding sequence

$$
T_i = \operatorname{MLP}_{time}(R_i) \in \mathbb{R}^{N_i \times 32}.
$$

### Concatenated classifier sequence

$$
X_i^{cat} = [E_i \;|\; \Delta E_i \;|\; T_i] \in \mathbb{R}^{N_i \times 832}.
$$

### Projected classifier tokens

$$
X_i = \operatorname{Proj}(X_i^{cat}) \in \mathbb{R}^{N_i \times 256}.
$$

### Time-decay gates

$$
\Gamma_i = \operatorname{DecayNet}(T_i) \in \mathbb{R}^{N_i \times 256}.
$$

### Recurrent hidden states

$$
H_i = [h_{i,1}, \dots, h_{i,N_i}] \in \mathbb{R}^{N_i \times 256}.
$$

### Output features

$$
O_i = [o_{i,1}, \dots, o_{i,N_i}] \in \mathbb{R}^{N_i \times 512}.
$$

### Binary logits

$$
L_i^{bin} \in \mathbb{R}^{N_i \times 1}.
$$

### Binary probabilities

$$
P_i^{bin} \in \mathbb{R}^{N_i \times 1}.
$$

---

## 10.3 Batched version

After padding to $N_{\max}$:

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

---

# 11. Why this classifier architecture is the right one

This architecture is the right first classifier because it uses exactly the information that matters for online detection:

* **current segment state** via $e_{i,j}$,
* **change of state** via $\Delta e_{i,j}$,
* **where in labor we are** via $\tau_{i,j}$,
* **how much time elapsed since the last segment** via $\Delta \tau_{i,j}$,
* **persistent accumulated evidence** via the recurrent hidden state $h_{i,j}$,
* **irregular segment spacing** via the decay gate $\gamma_{i,j}$.

This is much better than:

* an MLP on each segment independently,
* a static classifier on the full trajectory,
* or a recurrent model that ignores timing irregularity.

---

# 12. Online inference behavior

At test time, the model behaves as follows.

After each newly available segment $j$:

1. compute its pretrained segment embedding $e_j$,
2. compute $\Delta e_j$,
3. compute time features $r_j$,
4. compute projected token $x_j$,
5. decay previous hidden state using $\Delta \tau_j$,
6. update the GRU hidden state,
7. output:

   * current unhealthy risk $p_j^{bin}$,

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
o_j = [h_j \;|\; x_j] \in \mathbb{R}^{512}
$$

### Heads

* binary unhealthy head:
  $$
  \mathbb{R}^{512} \to \mathbb{R}^{1}
  $$

That is the architecture I would implement first.

If you want, I can next write this as a **complete PyTorch-style module specification** with class definitions, `forward()` signatures, and exact code-level tensor shapes.
