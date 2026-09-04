# `lag_attn_cfs` — the as-built design record

The causal-feature-domain lag-attention VAE-TEB: what it is, what it consumes, what it returns, what
it optimises, what it measures, and every place the built package differs from the design it was
built from.

Companion documents, none of them restated here: `teb_vae/lag_attn_rws/DESIGN.md` records the
architecture all six models share and `teb_vae/lag_attn_rws/model_explained.md` the latent
factorisation; `teb_vae/lag_attn_fs/DESIGN.md` records the feature target, the smear argument and
the four added gap readouts this package inherits; `hdf5_dataset/dataset_explained_research.md`
section 8.1 defines the causal dataset variant and
`hdf5_dataset/CAUSAL_SCATTERING_PHASE_HARMONIC_MATH.md` carries the warm-up and group-delay
mathematics. `RESULTS.md` in this directory carries the pre-registered criteria and the
measurements.

**What this document is for.** Reading it should leave a reader able to say what a reported number
of this model means and what it does not — which is a different question from how the architecture
works, and the one a model whose inputs are *differently produced* rather than differently shaped
makes easy to get wrong.

---

## 1. What the model is

The fifth cell of an encoder-by-target grid, and the first whose inputs do not contain their own
future:

```
                          conv-LSTM encoders          conv-Transformer encoders
  raw FHR target          lag_attn_rws                lag_attn_transformer_rws
  two-sided feature       lag_attn_fs                 lag_attn_transformer_fs
  causal feature          lag_attn_cfs   <- this      lag_attn_transformer_cfs
```

At each admitted 4-second anchor $t$ the model forecasts the next **two minutes of the stored
one-sided FHR feature future** — $H \cdot C_{\mathrm{keep}} = 30 \times 98 = 2940$ coefficients at
the shipped warm-up budget — twice: once from a target-only latent and once from a
source-conditioned one, through one shared decoder invoked twice under one noise draw. The gap
between the two forecasts, and the KL between the two latents resolved across lags, are the coupling
readout.

Structurally this is `lag_attn_fs` with one thing changed: which transform produced the features it
reads and forecasts. The two-sided cells read coefficients that at decimated step $t$ average raw
samples on **both** sides of $t$ — up to $965$ s into $t$'s own future on the slowest channel — which
is why their coupling readout is named `source_conditioned_kl_raw` and deliberately not called a
transfer entropy. The causal variant runs the same cascade through a strictly one-sided gammatone
bank matched to the production Morlets at half power, so a coefficient at $t$ is a function of
$\{x(s) : s \le t\}$ and of nothing else.

**What this cell can claim that no other cell can**, and the only thing it claims: its coupling
readout is measured on inputs that do not contain their own future. That is a property of the
*inputs* rather than of any number the run produces.

**It is an experiment, not a remedy.** `lag_attn_rws/PRIOR_SCALE_AND_PREDICTIVE_GAP.md` §5
establishes that the held-out predictive gain is negative because the source pathway does not
generalise — a failure in the source encoder, the lag attention and the posterior fusion, none of
which a target-domain change touches. This model is expected to reproduce it, probably more
strongly, because causal features are strictly weaker than two-sided ones. **The sign of `pred_gap`
is a criterion nowhere in this document.**

The whole model is

```python
class SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws): ...
```

with **a constructor and nothing else** — `vars(SeqVaeLagAttnCfs)` carries `__init__` and no other
callable — linearising as
`SeqVaeLagAttnCfs -> CausalWarmupInputs -> CausalFeatureForecastTarget -> FeatureForecastTarget ->
SeqVaeLagAttnRws -> Module -> object`. §6 records why the constructor is the one exception.

At the shipped configuration the model holds **4,655,987 parameters**, against **4,284,556** for the
encoder-axis comparison `lag_attn_transformer_cfs`. With every architecture switch of §17 at its
off-state and the single alignment clock restored it holds **5,146,334**, which is bitwise the model
that shipped before this revision and is the row the target-axis comparison against `lag_attn_fs`
(**5,126,326**) and the pre-revision encoder-axis comparison (**5,054,992**) are read on. §13
carries the arithmetic and decomposes every delta, and states the unaligned arm the channel
alignment (§3) moved this cell off.

**Six mechanisms of this cell are configuration rather than architecture**, each gated by a key
whose off-state reproduces the pre-revision model bitwise: where the lag attention's keys and values
come from (§8), whether the prior is told the source's arrival clock (§8), whether the decoder
carries a target-only persistence residual (§5), whether the reconstruction weights the horizon axis
(§5), how the lag bias is seeded (§8), and whether the source stream is aligned onto its own faster
clock (§3). §17 is the inventory, with the shipped value and the off-state of each.

## 2. Input contract

Read from the **causal** HDF5 shards through `train/data_module.py::GraphDataModule`, at
`trim_minutes: 1.0`. The field names and dtypes are the two-sided variant's exactly; only the widths
and the transform differ, which is what makes the driver's `transform` refusal load-bearing rather
than decorative.

| Field | Shape | Role |
| --- | --- | --- |
| `fhr_st`, `fhr_ph` | $(B, 300, 36)$, $(B, 300, 66)$ | target stream, concatenated to $(B, 300, 102)$ — **both the network input and the reconstruction target** |
| `up_st`, `up_ph` | $(B, 300, 36)$, $(B, 300, 15)$ | source stream, concatenated to $(B, 300, 51)$ |
| `weight` | $(B, 300)$ | per-step validity on the decimated grid |
| `fhr`, `up` | $(B, 4800)$ | raw traces; **not read by the model** — row 1 of the diagnostic page only |
| `guid`, `epoch` | — | figure titles, run provenance, **and the tile phase** (§4) |

$c_y = 36 + 66 = 102$ and $c_u = 36 + 15 = 51$, against the two-sided $109$ and $58$. Seven
scattering channels per block were dropped at write time because their one-sided warm-up outruns the
$330$-step stored segment at every trim; both phase blocks keep their full width, their $0.008$ Hz
band floor excluding those filters entirely.

**`fhr_st` and `fhr_ph` must be in `normalize_fields`.** They are this model's reconstruction target,
so an unnormalised block makes the Gaussian NLL meaningless against a unit-scale variance model, with
the loader raising nothing. The shared entry point refuses a config omitting either, field by field,
from `LagAttnCfsTrainer.TARGET_FIELDS`.

**`guid` and `epoch` must be in `load_fields`**, and that is the one difference from the two-sided
cells' list. `load_fields` is honoured literally with no forced additions, and the tile phase is
keyed on the pair; without them every segment of every recording is decoded at one tile grid forever,
with $A_{\max}$ a geometry constant either way, so no shape, no count and no metric would differ.
`trainer.preflight` refuses it.

**`fhr_up_ph` is absent from the variant and must stay absent from every config.** A coefficient
mixing both signals would put the source's own signal into the forecast target and destroy the
target-only / source-conditioned separation the coupling readout rests on. In `load_fields` the
loader would raise, but only after every rank had initialised; in `normalize_fields` it is silently
ignored and reads as though the block were being handled.

**Three properties of the stored data are easy to misread.** The coefficients inside $[0, W'_c)$ are
real float values, **not zeroed and not NaN** — the writer attaches the boundary as an attribute and
leaves the array untouched, so a consumer that ignores the attribute trains on values the
normalisation constants were accumulated *excluding*, and which are therefore on no defined scale.
And `weight`, the binary validity mask, is entirely independent of the warm-up: nothing anywhere
combines them.

## 3. Geometry: the budget and the floor are one decision

`TrimmedRawGeometry` is reused unchanged:

$$T = 300, \qquad H = 30, \qquad T_{\mathrm{valid}} = T - H = 270, \qquad F = 134.$$

A forecast at anchor $t$, horizon step $\tau$, reads target time $t + 1 + \tau$, and channel $c$ is
honest there only from $W'_c$ onwards. Requiring **every kept channel valid across every anchor's
whole window** gives one inequality,

$$t + 1 \ge W'_c \quad \forall\, c \in \mathrm{kept}, \; \forall\, t \ge F,$$

satisfied exactly when the budget $B = \max_{c \in \mathrm{kept}} W'_c$ and the floor $F \ge B - 1$.

**A second requirement binds the same floor, and under the shipped alignment it is the one that
decides.** Every one-sided filter carries its own composed group delay, spanning $13.3$ s to
$791.0$ s across a stored block, and the encoder reads a stream's channels as one vector per step —
which asserts that its entries describe one instant when they span thirteen minutes. The model
therefore re-indexes each channel onto a common clock, gathering channel $c$ with a shift of

$$d_c = \operatorname{round}\!\Bigl(\kappa\,\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr) \ge 0,
\qquad \tau_{\mathrm{ref}} = \max_{c \in \mathrm{kept\ target}} \tau_c = 402.1604\ \mathrm{s},
\qquad \Delta = 4\ \mathrm{s},$$

resolved from the shards' own `causal_delay_s` by `causal_warmup.resolve_warmup_budget` and never
from a YAML literal.

**$\kappa$ is there because `causal_delay_s` ships the wrong one of two delays for this purpose,
and the right one for every other.** The stored attribute is the gammatone envelope *mean*
$\tau_g = \gamma/(2\pi b)$, which is what a channel's staleness should be *reported* as. What a
gather has to cancel is the delay the channel's own spectrum actually realises, which is the
impulse response's energy centroid $(2\gamma - 1)/(4\pi b)$ — a fixed fraction of the mean,

$$\kappa \;=\; \frac{2\gamma - 1}{2\gamma} \;=\; 1 - \frac{1}{2\gamma} \;=\; 0.875
\quad\text{at the shipped } \gamma = 4,$$

shipped as `ALIGNMENT_DELAY_FACTOR` in `causal_warmup.py`. The discrepancy is one-sided rather
than symmetric jitter, and the reason is structural: $\tau_g(\nu) = \tau_g(\xi)\,b^2/(b^2 +
(\nu-\xi)^2)$ is *maximal* at the centre frequency, so a channel's own passband can only pull the
realised lag **down**, never up. Measured on $30$ segments of the aligned shard — each causal
`fhr_st` channel cross-correlated against the same channel of a centred bank — the median realised
delay is $0.9032\,\tau_g$ over all $30$ resolved channels and $0.882\,\tau_g$ over the nine slow
ones where the $4$ s grid quantises by under $2.5\%$, against $0.875$ predicted by the centroid and
$1.000$ predicted by $\tau_g$ itself. Every channel resolved at $r \ge 0.94$.

> **Two different $\kappa$ live in the bank's algebra and they are unrelated.** The wavelet's
> *zero-mean* correction $\kappa_k = \bigl(b_k/(b_k - i\xi_k)\bigr)^{\gamma}$ — the constant that
> makes $\tilde\psi_k[\tau] = a_k[\tau]\bigl(e^{\,i2\pi\xi_k\tau} - \kappa_k\bigr)$ integrate to
> zero — is per filter and complex, always carries its subscript, and perturbs $\tau_g$ by under
> $1\%$ ($+0.94\%$ on the slowest filter of the linear tail, below $10^{-5}$ everywhere else). The
> unsubscripted $\kappa = 1 - 1/(2\gamma)$ above is the single real **alignment** factor every
> alignment consumer scales by, and it is what this document, `ALIGNMENT_DELAY_FACTOR` and the
> preprint's `\kappa` mean by $\kappa$ everywhere.

$\kappa$ scales only the **difference** $\tau_{\mathrm{ref}} - \tau_c$, so the reference channel
still takes shift $0$ and the keep-index does not move at all: $\tau_c \le \tau_{\mathrm{ref}}$ is
scale-invariant, and the same $98/102$ and $47/51$ survive as before the factor existed. What it
moves is the shift *magnitudes*, $d_c \in [0, 85]$ where the unscaled rule gave $[0, 97]$.

### 3.1 Two references, not one: the source has its own clock

**The two streams are aligned onto two different references, and the offset between them travels
explicitly.** `causal_align_reference` fixes the target's clock; `causal_align_reference_source`
fixes the source's, resolved and snapped against the **source** stream's own stored delays and
driving the source `_align_stream` call alone. Both are task-level keys — they are resolution, not
constructor arguments — and the second defaults `null`, where the resolver's behaviour is
byte-for-byte the single-reference one it had before the key existed.

**Why a second key rather than a faster float in the first.** `resolve_warmup_budget` resolves one
reference *from the target stream* and applies it to both, and `_resolve_reference_delay` snaps an
explicit float against the kept **target** delays — so a faster value in `causal_align_reference`
would re-align the target stream at the source's clock and drop every slower target channel. That
is a change to the scored stream, and it is not the change wanted. The rejection recorded at
`causal_warmup.py`'s per-stream note stands and is narrowed rather than deleted: what it rejected
was per-stream *max* references, which restore an unknown inter-stream bias to the lag axis. A
deliberately **chosen** source reference whose offset against the target's is a single known number,
carried through the physical-lag arithmetic, the preflight causality record and the evaluation
console, is a different object.

$$\tau^y_{\mathrm{ref}} = 402.1604\ \mathrm{s}, \qquad
\tau^u_{\mathrm{ref}} = 288.2672\ \mathrm{s}, \qquad
\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}} = -113.8932\ \mathrm{s}.$$

**Where the shipped source value comes from, and what it was priced against.** It is not chosen
here; it is the unique candidate — drawn from the source stream's own stored delays — satisfying
both criteria of the tradeoff `warmup_budget.py` prints. A $20$–$60$ s physiological delay must land
**mid-window at every horizon step**: under the shipped pair it occupies lags $8.5$–$47.5$ of
$[0, 90]$, clear of both censoring edges, against $-20.0$–$19.0$ under the single reference, which
is censored below lag $0$ for $h > 9$. And at least $8$ of the $15$ `up_ph` channels must survive:
$9$ do, where the next faster candidate ($244.5197$ s) keeps only $6$ and is refused on that
criterion alone. What it buys is recency — the freshest surviving source channel moves from $402.2$
s to $288.3$ s of reported staleness before the anchor, $351.9$ s to $252.2$ s realised at the
energy centroid. What it costs is eight source channels, all of them the slowest harmonic siblings.

**The shipped shift vectors**, resolved from the aligned dev shard at the pair above. They are
published here rather than left to a re-derivation because they are the cheapest possible check that
a rebuild of the rule is the one the cells run and not a lookalike: an off-by-one in the trim, the
budget, either reference or the rounding moves them.

```
fhr_st  d_c = [85 85 85 84 84 84 84 84 83 83 83 82 82 81 81 80
               79 78 76 74 72 70 67 64 60 55 49 43 34 25 14  0]
fhr_ph  d_c = [83 83 83 83 82 82 82 82 82 81 81 81 81 81 81 80 80 80 79 79 79 78 78
               78 76 76 76 74 74 74 72 72 72 70 70 70 67 67 67 64 64 64 60 60 60 55
               55 55 49 49 49 43 43 43 34 34 34 25 25 25 14 14 14  0  0  0]
up_st   d_c = [60 60 60 59 59 59 59 59 59 58 58 57 57 56 56 55
               54 53 51 50 48 45 42 39 35 30 24 18 10  0]
up_ph   d_c = [30 24 18 18 10 10  0  0  0]
```

The two target vectors are the single-reference scheme's exactly — the target stream is untouched,
field for field, by the source key. The two source vectors are shorter because the source keep-index
is shorter, and smaller because the reference they cancel against is $113.9$ s nearer.

The worst quantisation residual over both streams is $1.9865$ s, measured against the **scaled**
difference $\kappa(\tau_{\mathrm{ref}} - \tau_c)$ and inside the $\Delta/2 = 2$ s the rounding
guarantees. $\min_c(W'_c + d_c) = 80$ on the target and $55$ on the source, where the unscaled rule
gave $91$ — a shorter shift is also a warmer grid.

A shifted channel is honest only from $W'_c + d_c$, so the floor must also
clear $\max_c(W'_c + d_c)$ on **both** input streams:

$$F \;\ge\; \max\Bigl(\underbrace{B - 1}_{\text{scored target}},\;
\underbrace{\max_c\bigl(W'_c + d_c\bigr)}_{\text{shifted inputs}}\Bigr).$$

The second term costs no warm-up beyond the reference channel's own. With $W_c \approx \rho\tau_c$,
$W_c + \Delta d_c = \kappa\tau_{\mathrm{ref}} + (\rho - \kappa)\tau_c \le \rho\tau_{\mathrm{ref}}$
— because $\rho \approx 1.5 > \kappa$ and $\tau_c \le \tau_{\mathrm{ref}}$ — with
equality at the reference, where the shift is zero. Measured, $\max_c(W'_c + d_c) = 134 = B$ exactly
on the **target** stream, and the factor does not disturb that: it shrinks every shift and leaves
the reference's alone, so the maximum is attained at the same channel and at the same value. On the
**source** stream the same maximum is $92$, because §3.1's nearer source reference is what its
shifts cancel against; the source therefore never binds the floor, and the target's requirement
decides alone. So the shipped values are $B = 134$ and $F = \max(133, 134) = 134$: the common clock
costs exactly one anchor, $137 \to 136$, and never more. The constructor refuses a lower floor by name and says which
of the two requirements it violated: below the first the objective scores the assumed pre-recording
history of the slowest kept channel as signal, and below the second the encoder reads a shifted
channel inside its own warm-up — with every shape correct and every warm-fraction readout still
reporting $1.0$.

**What the alignment costs instead is recency and twelve source channels.** Every source channel
slower than its own reference is dropped, $51 \to 39$ (`up_st` $30/36$, `up_ph` $9/15$), because
aligning one would need a *negative* shift — reading it from a later stored step, i.e. from raw
signal after the anchor — which is a correctness requirement rather than a warm-up policy. Under the
single reference the drop was $51 \to 47$ with all fifteen `up_ph` surviving; §3.1 prices the trade
and pins the value. And the freshest uterine activity any channel reports moves from $13.3$ s of
reported staleness to its own reference's $288.3$ s before the anchor — whose *realised* value is
$\kappa\tau^u_{\mathrm{ref}} = 252.2$ s, since an aligned channel at step $t$ carries raw signal
centred at $t - \kappa\tau^u_{\mathrm{ref}}$ whatever its own $\tau_c$. `causal_align_reference:
null` is the unaligned arm and is bitwise the model that shipped before either key existed;
`causal_align_reference_source: null` is the single-clock arm, which is the alignment as it shipped
before §3.1. Both are named sweep arms rather than reachable-in-principle settings.

Measured against the committed causal fixture, the budget keeps **98 of 102** target channels —
`fhr_st` $32/36$ and `fhr_ph` $66/66$ — dropping the four `fhr_st` channels at
$W' \in \{162, 194, 233, 278\}$. $134$ is where `fhr_ph` tops out, so all $66$ phase channels
survive, and the `fhr_st` staircase has a channel at exactly $134$ with the next at $162$: the
boundary lands on the $\approx 0.008$ Hz frequency edge both phase selections already use rather than
on an arbitrary cut.

**The floor is a causal-validity boundary, not a clinical policy.** $134$ steps is $536$ s, i.e.
$8.93$ minutes. Nothing physiological distinguishes it from step $150$, and the model's own $30$-step
warm-up has long passed at either; at $H = 30$ a floor of $150$ keeps the **identical** $98$ channels
and costs one tile and $16$ covered target steps. If "ten minutes of observation before a forecast"
is meant as a scientific policy rather than as a filter-validity statement, `sweep_floor_150.yaml`
ships it as a one-key change — the pairing requires $F \ge B - 1$, not equality.

**The mask stays three-dimensional, and that is what the pairing buys.** The alternative is a
per-$(\text{anchor}, \tau, c)$ validity mask using the partially-valid coefficients at low anchors.
It breaks `compute_loss` in fourteen places, thirteen loudly and **one silently**:
`contributing_anchors` reduces only `dim=-1`, so on a 4-D mask it returns $(B, A, H)$ and every
denominator inflates by $H$ with no exception. More fundamentally every denominator of a *loss term*
is an anchor count rather than an element count, so a valid-channel count that varied per anchor
would make nats-per-anchor shrink silently with mask density — and that density varies systematically
with $t$, so late anchors would dominate. Under the pairing the mask stays $(B, A, H)$ and broadcasts
over channels exactly as in every sibling, the block width is the constant $2940$, and $\beta$ keeps
its meaning.

**Shortening the horizon buys back channels.** Anchors live in $[F, T_{\mathrm{valid}})$, so a
shorter horizon lengthens $T_{\mathrm{valid}}$: the shipped $H = 30$ admits $136$ anchors where
$H = 15$ admitted $151$, and at the budget that would keep all $102$ channels ($W' = 278$) there are
$8$ anchors at $H = 15$ and none at all at $H = 30$ — so keeping every channel is geometrically
impossible at the shipped horizon. The horizon is therefore one of the two levers on the warm-up
cost rather than an independent preference, which is why `sweep_horizon_15.yaml` ships the shorter
horizon as an arm: it is where the anchor cost of the two-minute forecast is measured.

## 4. Anchor tiling

Every sibling decodes all anchors in $[0, T_{\mathrm{valid}})$ every step, so consecutive windows
overlap $(H-1)/H$ and each target coefficient is scored by up to $H$ anchors. This model decodes a
**tiled** set,

$$\mathcal{A}(\varphi) = \{\, F + \varphi + kS \;:\; k \ge 0,\; F + \varphi + kS < T_{\mathrm{valid}} \,\},
\qquad S = H = 30,$$

so windows partition the timeline and no target coefficient is scored twice in one step; over epochs
every anchor in $[F, T_{\mathrm{valid}})$ is used, because the phase rotates with the epoch. At the
shipped geometry that is $5$ tiles for $\varphi \le 15$ and $4$ otherwise, mean exactly
$136/30 = 4.533$.

**Validation and test decode every valid anchor**, at stride $1$ and phase $0$. A single phase is
deterministic but *phase-biased*: it would sample the same $5$ of $136$ anchors at a fixed offset
from the segment start forever, so any structure varying with position in the segment would never be
seen at another offset. There is no gradient at either stage, so neither the redundancy argument nor
the memory argument that motivates the tiling applies.

**The stride and the phase are forward arguments, not mode-derived.** The signature is
`forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)`, and the task supplies both
from the `stage` string the step dispatcher already passes: $(\varphi_b, S)$ on `train`, $(0, 1)$ on
`val` and `test`. Resolving the stride from `self.training` was rejected: the diagnostic callback
calls `eval()` *during* training and then `compute_loss`, so the decoded anchor set — and therefore
`total_loss`, `nll_base_block` and `anchors_per_sample` — would be a function of the **dropout
switch**, in direct contradiction of that callback's own promise that a figure cannot quietly
disagree with the objective it illustrates. A bare-constructed `nn.Module` defaults to
`training=True`, so every test that forwarded without setting a mode would pin the training geometry
by accident. Two guards make the arguments hard to get wrong: an absent phase **raises** once the
resolved stride exceeds $1$, and a non-zero phase raises at stride $1$, where
$\mathcal{A}(\varphi)$ truncates rather than rotating and would silently drop $\varphi$ anchors.

**The phase is derived, never drawn.**

$$\varphi_b = \mathrm{blake2b}\big(\texttt{guid}_b \,\Vert\, \lfloor\texttt{domain\_start}_b\rfloor
\,\Vert\, \texttt{train\_epoch} \,\Vert\, \texttt{seed}\big) \bmod S$$

Three properties are load-bearing and each replaces something that does not work. `hashlib.blake2b`
rather than Python's `hash()`, which is salted per process by `PYTHONHASHSEED` and is therefore
stable neither across DDP ranks nor across a resume — and fails *silently*, since $A_{\max}$ is a
geometry constant either way, so no same-process test can see it. `domain_start` is in the key and
not only `guid`, because `guid` identifies the *recording*: an unshuffled loader over per-recording
shards puts consecutive segments of one recording in one batch, and keying on `guid` alone would
leave in place the within-batch gradient correlation the tiling exists to break. And a
`torch.randint` inside `forward` is not an option: it would consume the global RNG stream, move the
reparameterisation $\epsilon$ and break every bitwise comparison in the suite. A derived phase needs
no cross-rank collective, because each rank hashes its own samples.

**The count varies with $\varphi$, so the anchor tensor needs a validity companion.** $A_{\max} =
\lceil (T_{\mathrm{valid}} - F)/S \rceil$ is a **geometry constant** — $11$ at the shipped training
geometry and $136$ at the dense evaluation resolution — so no rank can disagree on shape. Short rows
**repeat their last valid anchor and mark it invalid**, and that is not cosmetic: a padded slot
holding a *distinct legal* anchor would produce a fully live `forecast_mask` row, so its target block
would be gathered and scored twice while `kl_mask`'s scatter deduplicated it — the reconstruction and
KL denominators would diverge and quietly change what $\beta$ means.

**Shape contract, stated once.** Everything on the anchor axis is *sparse* —
$(B, A_{\max}, \ldots)$ with `anchor_valid` carrying the padding — from the decoder's input through
the four forecast tensors, the target, `forecast_mask`, `contributing_anchors` and every per-anchor
metric. The **only** place the anchor axis is scattered back to dense is `kl_mask`, which must return
$(B, T)$ because the latent tensors it gates are $(B, T, d_z)$ and are produced at every step
regardless of which anchors are decoded.

The memory consequence is what makes the wider causal block affordable at training: the four forecast
tensors and the target go from $(B, 136, 30, 98)$ to $(B, 5, 30, 98)$, roughly $7.5$ MB rather than
$114$ MB **each** at $B = 128$. Validation pays the dense figure, which is what `lag_attn_fs` already
pays on every step of every stage.

`anchor_stride: 1` recovers the dense range exactly and ships as `sweep_anchor_stride_1.yaml`.

## 5. Loss, and what its nats are summed over

$$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
+ \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
+ \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
+ \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

in nats per anchor, computed by `lag_attn_rws/nets/losses.py` — **the same code, not a copy of it**,
reached through the inherited `compute_loss`. The anchor set travels to it through
`forward_outputs['anchor_index']` rather than through a subclass, so the seam is expressed once in
the shared objective and the shared masks; `CausalFeatureForecastTarget` overrides neither
`compute_loss` nor `_build_forecast_target`.

**`lambda_boundary` is refused at any non-zero value, and that refusal is this cell's own.** The
boundary-continuity term is a slicing identity over **adjacent** anchors — it joins anchor $t$'s
first forecast sample to anchor $t-1$'s last. This family always decodes a tiled set whose entries
are $S$ apart, so at any weight the term would join two windows a whole horizon apart and score the
gap between them as an error. The shared objective raises on the combination; the driver's pre-flight
moves the failure to before a run directory exists. The other two shape weights ship at $0.0$ for the
two-sided sibling's reason: they price the envelope and slope of a raw *waveform*, and this block's
last axis is $98$ surviving wavelet channels, where "adjacent" means "the next filter".

**`block_width` is $C_{\mathrm{keep}} = 98$, not `geometry.r`.** It feeds only `mean_logvar_full`,
`mean_logvar_base`, `logvar_full_floor_frac` and `logvar_full_ceil_frac` — **not the loss** — so
passing the raw grid's $R$ changes no loss, fails no shape check, and rescales exactly the four
diagnostics `logvar_clamp` is re-derived from.

### The reconstruction is weighted per stored block

Both reconstruction terms carry a per-channel weight $w_c$, resolved at construction from
`target_weight_st` and `target_weight_ph` and registered as a non-persistent buffer positional
against `target_keep_index`. It ships at $(1.0, 0.1)$, and the conv-Transformer cell ships the same
pair — this is a property of the objective, not of an encoder, so a difference across that edge
would compare two architectures under two losses.

**Uniform was never neutral.** At the shipped budget $66$ of the $98$ survivors are phase-harmonic
against $32$ scattering, so an unweighted objective spent $67.3\%$ of itself on `fhr_ph` by nothing
more deliberate than channel count. The shipped pair resolves to $2.5389$ and $0.25389$ and moves
scattering's share to $82.9\%$. It is a hypothesis rather than a settled fact: the diagnosis of the
first conv-Transformer run measured that the channels the model abandons are largely the fast
phase-harmonic ones, whose persistence $R^2$ is negative from horizon step $4$, and the two-minute
horizon adds fifteen further steps in that regime.

**The vector is renormalised so $\sum_c w_c = C_{\mathrm{keep}}$**, which is what keeps the rest of
the configuration valid: the weighted block sums to the same magnitude the uniform one did, so
`gradient_clip_val` and `additive_margin` are not invalidated a second time and $\beta$ keeps its
standing against the reconstruction. Only the *distribution* moves, so what the configuration states
is a **ratio** — $(10.0, 1.0)$ describes the same objective.

At $(1.0, 1.0)$ the resolved vector is exactly ones and `raw_sample_score` skips the multiplication
entirely, so an unweighted cell scores a block that is **bitwise** the one it scored before the
mechanism existed. `tests/test_metrics.py` asserts that rather than approximating it, because every
other cell of the grid depends on it.

**What it costs is a unit rather than a scale.** A weighted reconstruction is not a log-density, so
$\beta = 1$ is no longer the exact ELBO. **The evaluation pipeline is deliberately left
unweighted** — `eval/` reduces through the same functions with no vector, so its `nll_*` and
`pred_gap` remain true log-densities and `pred_gap_mc_likelihood_pct` remains a probability
statement. Training optimises the weighted objective and the evaluation measures the unweighted one;
a `pred_gap` from `metrics_history.csv` and one from `summary.json` are therefore no longer the same
quantity.

The four channel-axis splits (`pred_gap_st`, `pred_gap_ph`, the three warm-up tertiles) apply the
same vector, so each remains a partial sum of the `pred_gap` printed beside it. Reading
`pred_gap_st` against `pred_gap_ph` is what judges the choice.

### The reconstruction is also weighted per horizon step

A second weight, on the other axis of the block, and it exists for a measured reason: at a uniform
horizon weighting the objective is paid the same for step $29$ as for step $0$, and the channels the
model abandons under that pressure are the fast ones, whose persistence $R^2$ is already negative by
horizon step $4$. Doubling the horizon from $15$ to $30$ added fifteen further steps in exactly that
regime. `horizon_weight_halflife_steps` resolves a geometric decay,

$$w_\tau \;\propto\; 2^{-\tau/\lambda}, \qquad
\sum_{\tau} w_\tau = H, \qquad \lambda = 15.0 \text{ shipped},$$

registered as a non-persistent buffer and threaded through the objective **exactly as
`channel_weight` already is** — `raw_sample_score` -> `masked_raw_block_per_anchor` ->
`masked_raw_likelihood` -> `compute_loss` — applied on the horizon axis. The two weights compose as
$w_c w_\tau$ and neither is aware of the other. At the shipped $\lambda = 15.0$ and $H = 30$ the
resolved vector runs $1.8063$ at $\tau = 0$ to $0.4729$ at $\tau = 29$, a $3.8\times$ spread.

**The renormalisation to $\sum_\tau w_\tau = H$ is what keeps the rest of the configuration valid**,
for the same reason the channel weight is renormalised to $C_{\mathrm{keep}}$: without it the block
would have shrunk by $1.81\times$ against an unmoved KL, and `gradient_clip_val`, `additive_margin`
and $\beta$'s standing against the reconstruction would all have gone out of date silently. Only the
*distribution* over horizon steps moves. `null` — the off-state — registers no buffer at all, and
the delegation sites read `None`, so an unweighted cell scores a block that is **bitwise** the one
it scored before the mechanism existed. A vector of the wrong length, or one shaped $(H, 1)$, is
refused by name rather than broadcast onto the channel axis.

**What it costs is the same unit the channel weight costs.** A block score weighted on either axis
is no longer a log-density, so $\beta = 1$ is not the exact ELBO — the caution stated for
`channel_weight` above applies unchanged and is restated where this weight lands. **The evaluation
pipeline applies neither weight**, deliberately, so its `nll_*` and `pred_gap` remain true log
densities. A training-path `pred_gap` and an evaluation `pred_gap` were already different quantities
under the channel weight; they now differ on a second axis as well. `pred_gap_tau_first`,
`pred_gap_tau_last`, `pred_gap_st`, `pred_gap_ph` and the six warm/novelty splits are all computed
under **both** weights, because they are partial sums of the `pred_gap` printed beside them and the
objective applies both to it: `pred_gap_st + pred_gap_ph - pred_gap` is exactly $0.0$ under the
weighted objective, which is the identity `check_run.py`'s fourth criterion reads.

### The decoder mean carries a target-only persistence residual

$$\mu[t, \tau, c] \;=\; w[\tau, c]\, y[t, c] \;+\; f_\theta(z_t)[\tau, c].$$

`w` is a learnable $(H, C_{\mathrm{keep}})$ **raw `nn.Parameter`** — raw rather than a module, which
is what makes it survive the generic `initialization(self)` pass the way `lag_score_bias` already
does — seeded from a fixed geometric decay in $\tau$. It applies to the **mean head only**; the
log-variance path is untouched. `persistence_residual: false` does not build the parameter at all,
and the decoder is then bitwise the one that predates the mechanism.

**It opens no source bypass, and that is a property of what is fed to it rather than of the weight.**
$y[t, c]$ is the target's own stored vector **at the anchor step**, gathered from the target stream
alone, which is input to both branches already; and **both** decoder invocations receive the *same*
tensor, so $\mu^{\mathrm{full}} - \mu^{\mathrm{base}}$ is exactly the residual-free gap and
`pred_gap` stays a pure source readout. The permutation control's re-decode carries the same matched
tensor for the same reason — the control permutes the *source*, and a decode without the residual
would shift the shuffle gap for a non-source reason — and the evaluation's Monte Carlo estimator
decodes every branch and every draw through the same one, since the residual is a property of the
anchor rather than of the latent.

**At the anchor step rather than at the last valid step**, and that is the training forward's own
definition: no validity weight enters the forward, anchors are target-warm by the pairing refusal of
§3, and an anchor invalid at its own step is fully masked in the loss, so the residual scores nothing
on it. The gather is taken **before** the gate's shift, over the keep-index alone.

The raw-target cells do not get it, deliberately; §14 records why.

### What the nats are, and are not, comparable to

- **Not comparable across the target axis.** The reconstruction sums $2940$ coefficients against
  `lag_attn_fs`'s $2340$ and the raw cells' $H \cdot R$ samples, so the block is not the same. The
  *horizon* now is: every cell of the grid forecasts $30$ steps, so the forecast question is shared
  and a per-horizon-step reading is made along one axis. Nothing restores comparability of the
  block, because $C_{\mathrm{keep}}$ is what the warm-up budget decides.
- **Not comparable across warm-up budgets within this model.** $C_{\mathrm{keep}}$ is what the budget
  decides, hence the decoder width, hence the block every nat is summed over. Two arms of this model
  at two budgets have non-comparable `pred_gap` and **mutually unloadable checkpoints** — and the
  class stamp cannot separate them, since both stamp `SeqVaeLagAttnCfs`. Only the width the stamped
  `target_keep_index` implies does.

## 6. Two mixins, one architecture, one constructor

The target domain is split in two, and the split is what lets a second architecture compose it:

| Mixin | What it owns |
| --- | --- |
| `CausalWarmupInputs` | the input warm-up mask, the lag validity floor, the tiled anchor set and the forward that decodes at it |
| `CausalFeatureForecastTarget` | the one-sided channel layout, the budget-and-floor refusal and the resolved readouts of §10 |

**Neither names an encoder.** Both read only attributes every architecture's constructor sets before
building its decoder, which is what lets `lag_attn_transformer_cfs` compose exactly these two objects
over a different history stack. `CausalFeatureForecastTarget` subclasses
`lag_attn_fs`'s `FeatureForecastTarget` and overrides **two** of its members —
`TARGET_BLOCK_SPLIT` and `_resolved_forecast_gaps`; `_default_decoder_out_channels`, `compute_loss`
and `_build_forecast_target` are inherited verbatim.

**The order of the bases is load-bearing.** The mixins come first, which is what makes the tiled
forward, the warm-up adapter, the floored lag mask, the width hook and the resolved gaps win method
resolution over the raw-target ones. Reversed, the decoder is built at $R = 16$ and a $98$-channel
block is scored against it — a failure that is loud but not where a reader would look for it:
`block_width` would not catch it, since it feeds only the four log-variance diagnostics and no shape
check, while `raw_sample_score` computes $(\text{target} - \mu)^2$ on $(B, A, H, 98)$ against
$(B, A, H, 16)$, which is not broadcastable.

**The constructor is the one member of the class, and only because of the driver.**
`trainer._build_model_kwargs` builds a run's kwargs by sweeping
`inspect.signature(MODEL_CLS.__init__)`, so a `**kwargs` signature would forward four keys and
silently build an all-defaults model. The signature is the base's written out in full, with
`target_delays` and `source_delays` **removed** and four keywords put in their place —
`target_warmup_steps`, `source_warmup_steps`, `anchor_stride` and `lag_floor`. Removing the two is
the point: a warm-up is a leading *mask* and `ChannelDelay` is a *shift*
($\mathrm{out}[t,c] = x[t - \delta_c, c]$), so a warm-up routed under a delay name would train a
different model with every shape intact — and a checkpoint's `model_kwargs` would be ambiguous
between two families under one key name.

## 7. The warm-up is a mask, and it lives inside the availability adapter

The stored coefficients inside $[0, W'_c)$ are real values normalised with constants that excluded
exactly that region. They are zeroed after normalisation and the fact is carried by the availability
channel, and **both happen inside `AvailabilityInputAdapter`**, which already holds the exact tensor:

$$m_{t,c} = \mathbb{1}[t \ge \delta_c], \qquad
e_t = W_x (x_t \odot m_t) + W_m (m_t - \mathbf{1}) + \mathbb{1}\Big[\textstyle\sum_c m_{t,c} = 0\Big] e_{\mathrm{start}}$$

The only change to the shipped adapter is the $\odot\, m_t$, and it is **provably inert for the four
two-sided models**: a gated model reaches the adapter through `ChannelGate.forward`, which returns
`gathered * available` with the same $\delta$, so those positions are already exactly zero; an
ungated model builds no availability buffer and no mask. A separate `ChannelWarmup` module was
rejected — it would make the mask vector and the announcement vector two copies of one quantity free
to disagree.

The model overrides `_build_adapter` to pass the resolved warm-up rather than reading
`gate.delay.delay_steps`, which under `ChannelGate(delays=None)` is all zeros: without the override
neither availability term would be built at all.

**The adapter is fed $W'_c + d_c$, not $W'_c$, and that is what brings both start embeddings into
existence.** A gathered-and-shifted channel is clean exactly when the step index has reached *both*
the warm-up and the shift, so `_build_adapter` passes the elementwise sum; feeding the warm-up alone
would announce a channel warm by up to $85$ steps before it is, with no crash and no metric moving.

$e_{\mathrm{start}}$ exists only when *every* channel of a stream waits at least one step. Unaligned,
both streams had a channel at $W' = 0$ — `fhr_ph` on the target and `up_st` on the source — so both
indicators were identically zero and neither vector was constructed. Aligned,
$\min_c(W'_c + d_c) = 80$ on both streams, so **both are built**: one learned vector of width
$d_{\mathrm{model}}$ per stream, $+2 \times 128$ in §13's table, and a live "everything here is still
pre-warm-up" token in the forward pass over the first $80$ steps of every segment. It is reached by
the leading steps of every segment rather than by none, which retires the
`find_unused_parameters=False` hazard that made this construction-time flip worth avoiding when
`use_up_st: false` was the only way to trigger it.

## 8. The source, and the constraint that has no clean solution

Lag attention searches $L = 91$ lags, so at anchor $F = 134$ it reads source states back to step
$44$. `up_ph`'s band is $0.008$ to $0.05$ Hz, so every one of its $15$ channels is built from slow
wavelets: minimum $W' = 41$, minimum staleness $150.8$ s. Unaligned, at step $44$ only $27$ of $51$
source channels were warm and just $1$ of $15$ `up_ph`; under the shipped alignment the combined
$W'_c + d_c$ has minimum $55$ on the source, so at step $44$ **none** of the $39$ kept channels is
warm. The compromise is real and it is smaller than it was: the mean per-channel warm fraction over
the whole $(\text{anchor}, \text{lag})$ grid is $0.885$ unaligned, $0.863$ under one shared clock,
and $0.970$ under the shipped source clock of §3.1 — buying back what the single reference paid,
because a source channel aligned onto a $113.9$ s nearer reference is honest $113.9$ s earlier.

Neither gating nor shortening fixes the remainder. A source budget of $44$ would make every
reachable lag fully warm at the price of $14$ of the $15$ `up_ph` channels; a `max_lag` of $16$ would
keep `up_ph` nearly whole but reduce the lag search to $64$ s, which does not cover the $20$ to
$120$ s contraction-to-deceleration delay the model exists to find. The tension is a $20$-minute
segment against a $6$-minute lag search against a $9$-minute warm-up, and it is a property of the
data.

So **the warm-up budget gates no source channel at all** — all $51$ survive it — and the twelve the
alignment removes are removed for the unrelated reason that they are slower than the source
reference and could reach it only by being read from a later stored step. $39$ are kept, `up_st`
$30/36$ and `up_ph` $9/15$; the availability mechanism announces per step when each arrives, and the
residual is made *measurable* rather than resolved: `lag_floor` generalises the lag validity mask
from $\mathbb{1}[t - \ell \ge 0]$ to $\mathbb{1}[t - \ell \ge F_u]$ and ships at $0$, where the mask
is bitwise the sibling's, and `source_lag_warmth_frac_st` / `_ph` report the attention mass landing
on lags where at least half of a stored source block's channels are past their warm-up. **A small
value there is the expected finding, not a failure.**

**Those two columns are only now able to say anything, and the reason is worth recording.** The
block-warm patterns behind them were built from $W'_c$ alone while the availability mask (§7) and
the anchor floor (§3) both used $W'_c + d_c$ — three sites, two of them enforcing and one of them
*reporting*, which is why nothing failed. Since $d_c \ge 0$ the bias was one-sided: the readout
could only ever report the source as **warmer** than it is, and on a single-reference configuration
that made `source_lag_warmth_frac_st` identically $1.0000$ for any attention distribution
whatsoever — a column that could not vary and therefore measured nothing. `CausalWarmupInputs` adds
the shift before splitting the stream into its two blocks, and the shipped geometry follows: `up_st`
reaches half-warmth at step $59$ and `up_ph` at step $86$, against $84$ and $117$ under one shared
clock. At the lowest anchor $F = 134$ the searched source steps span $44$ to $134$, so $76$ of the
$91$ lags are `up_st`-warm and $49$ are `up_ph`-warm, against $51$ and $18$ — which is the
compromise this section describes, now visible in the CSV instead of hidden behind a constant.

### 8.1 Where the keys and values come from, and why not from the deep encoder

**`lag_kv_source` decides which source representation the lag attention's keys *and* values are
built from**, and it is the seam that decides what a lag readout can possibly report.

| value | K/V representation | receptive field at the shipped schedule |
|---|---|---|
| `encoder` | the deep source history state $h_u$ | unbounded — the LSTM makes it a function of the whole prefix |
| `conv_stem` (shipped) | `CausalConvStem` over the availability adapter's output | $387$ steps at `encoder_extra_dilations: [8, 16]`, `encoder_extra_kernel: 15` |
| `adapter` | the availability adapter's output directly | $1$ step — per-step content, linear in the stored coefficients |

**The argument for localising, stated once.** Under `encoder` the key and the value at lag $0$ are a
deterministic function of every earlier source step, so by the data-processing inequality the lag-$0$
entry already contains whatever any later-lag entry carries. An attention distribution that puts all
its mass there is not reporting an absence of delay; it is reporting that the representation made
every other lag redundant. Nothing downstream can undo that, and no better-looking attention map
would mean anything different, which is why **both** K and V move rather than the keys alone:
localising keys while leaving deep values would leave the lag-$0$ value informationally dominant and
the degeneracy standing under a prettier map.

**Under a local arm the deep source encoder is not built at all.** Nothing else in the model
consumes $h_u$, so constructing one would be a starved parameter block in DDP's expectation set and
a claim in the manifest that the model attends over a state it does not have. Neither encoder class
in this family can be built *without* its deep stage — the recurrent one has no LSTM-free mode and
the conv-Transformer one refuses zero attention blocks, both by design — so `conv_stem` is a small
wrapper over the existing block classes with an output norm, and the shared encoders' own validation
is untouched. In this cell's parent the `causal_norm` branch (`causalize_norms`) follows the stem
wherever it followed the encoder.

**One thing the arm cannot do, and it is the resolution floor.** A local K/V bounds the *encoder's*
memory, not the transform's: a stored coefficient is a causal integral with its channel's own group
delay, so beneath the K/V receptive field sits a smearing no model-side choice reaches. §3.1's
reference is the part of that geometry can move; §14 records the rest.

**`source_state` is the tensor the attention actually reads**, under every arm, and both controls
follow it rather than the encoder: `perm_forward_outputs` permutes it and
`source_null_forward_outputs` re-encodes gate -> adapter -> the configured K/V path. Their
*signatures are frozen* — each resolves the path from model attributes — so no call site in the task
layer or the evaluation moved. Had they not followed, `source_specificity` and `kld_source_null`
would silently be probing a tensor nothing reads. Queries, the posterior fusion, `entmax15`, the
Shaw per-lag key bias, the frozen $W_o$ convention and the lag mask are untouched by the arm.

**`alibi_slope_scale` is the other half of what a lag profile can express**, and it is a
configuration of the existing bias rather than a new mechanism. `lag_bias_init: alibi_decay` seeds a
learnable $(\text{num\_heads}, L)$ parameter; the scale is what it is seeded *with*. The shipped
$0.0$ seeds it **flat**, so the profile the model reports is what training put there. At $1.0$ it is
seeded with a monotone decay towards lag $0$ — a named comparison arm, and the pair is what separates
"the data says lag 0" from "the initialisation says lag 0". `lag_bias_init: normal` is **not** the
flat arm: it builds no bias parameter at all.

### 8.2 The prior is told the arrival clock too, and what that can and cannot reach

**Logging is not enough, and the existing control does not cover this.** The source availability
pattern $m^u_{t,c}$ is a deterministic function of $t$, identical for every sample, and it enters
$q(z \mid Y, U)$ but not $p(z \mid Y)$ — so the posterior can be pushed off the prior by the
availability *clock* alone, inflating `source_conditioned_kl_raw` with no source information in it.
The permutation control cannot detect that: it deranges `source_state` across the batch, and **every
row carries the same availability pattern**, so no permutation of rows can remove something every row
shares.

The control that does isolate it is the **source-null arm**, `kld_source_null`. It cannot reuse
`perm_forward_outputs`, which starts from an already-encoded source state — permuting an encoded
state is equivalent to re-encoding a permuted stream, but a zeroed stream is not a permutation and
both the adapter and the K/V path are nonlinear, so the null re-runs `source_gate`, `source_adapter`
and **whichever module `lag_kv_source` selected** from a zeroed stream. That last clause is the
whole of why the control is still a control after §8.1: it re-encodes through the forward's own
path, so it probes the tensor the attention reads rather than one nothing consumes. It costs one
source *encode* and no decode: it is a KL
readout, so it needs only $(\mu^{q,\mathrm{null}}, \ell^{q,\mathrm{null}})$ and the existing
`kl_mask` support, and deliberately **no `torch.randn_like`**, which would shift the
reparameterisation stream for every subsequent step. With $x \equiv 0$ the adapter's output depends on
$m_t$ alone and is therefore identical for every batch element, so $h_u^{\mathrm{null}}$ is one
$(1, T, d_{\mathrm{model}})$ tensor broadcast over the batch; the query is still $\mu^p$, so the KL
still varies per sample.

**Zeroing is the correct null, and it floors slightly less than "the clock".** The causal shards'
normalisation constants were accumulated *excluding* the warm-up region, so zero is the channel mean
over the region the model actually reads — a per-batch or per-sample mean would leak the sample's own
source statistics. But the encoder's response to a flat trajectory is not literally the availability
pattern's response, so `kld_source_null` floors no source *variation*, which is a slightly weaker
statement than "the clock alone". `source_conditioned_kl_raw - kld_source_null` is the part
attributable to source variation; **if the two are equal, the coupling readout is measuring a
clock.**

**`prior_availability_input` makes the clock symmetric rather than subtracted.** The asymmetry above
is that a deterministic function of $t$ enters $q$ and not $p$; the mechanism gives the same function
to $p$, so the term cancels in the divergence instead of being measured out of it afterwards.
`FullLatentPriorHead.forward(h_y, clock=None)` takes it through the head's **own** `LayerNorm` and a
linear projection — deliberately *not* shared with the adapter's `mask_proj`, because the pattern is
what must be shared and the map is not; sharing the map would couple the two pathways' gradients.
The projection is zero-initialised **and re-zeroed in the parents' post-initialisation block**,
beside `_zero_init_delta_heads`, because both parents run a generic `initialization(self)` pass after
module construction that would otherwise xavier-refill a constructor-only zero. With the re-zero in
place the prior is bitwise the flag-off prior at initialisation, the exact-zero-KL start survives,
and `head_init_calibration`'s prior-scale pinning is untouched.

**What the prior is given is not the announcement.** The per-channel staircase
$\mathbb 1[t \ge W'_c + d_c]$ is **constant on every scored anchor**, and provably so:
`CausalFeatureForecastTarget._check_anchor_floor` refuses any floor below $\max_c(W'_c + d_c)$,
which is exactly the last step at which the staircase changes. So on every arm, every reference and
every budget this family admits, every scored anchor sees an all-ones announcement — and a constant
vector through a `LayerNorm` and a linear map is an offset the prior head's biases already span.
What survives past the floor is the source pathway's **memory** of the arrivals, which a constant
cannot carry, so the clock is `CausalWarmupInputs._prior_clock`: the encode of a **zeroed** source
stream through the configured K/V path, computed at batch $1$ and broadcast — the same tensor the
source-null control feeds the posterior. It varies at every scored anchor, carries zero information
about the source's *values* by construction, is **detached** so no gradient couples the prior to the
source pathway, and is **forced out of train mode** so dropout cannot make it a fresh draw per step
rather than a function of $t$. It costs one extra batch-$1$ source encode per forward.

**The invariant is restated, not weakened.** "The prior never sees the source" becomes **"the prior
sees no function of the source *values*"**, and that is the sentence the source-purity tests assert.
The two statements differ by exactly one object — an encode of silence — and nothing about the
forecast claim or the coupling readout rests on the stronger one.

**What the mechanism can reach, and what it cannot.** The clock is a function of $t$ that varies at
every scored anchor, so it is the right object; but its reach is bounded by the posterior's own
parameterisation. The posterior is a bounded residual on the prior, $\mu^q = \mu^p + \Delta_\mu$, so
$\mu^q - \mu^p$ *is* the delta head's output and the mean half of $\mathrm{KL}(q^\varnothing \Vert
p)$ is a function of the delta head alone — the prior's input does not appear in it at all.
Conditioning the prior can therefore reach only the **variance half**, through $\ell^p$'s appearance
in a ratio and a denominator. That is a property of the parameterisation rather than of the clock,
and it is why `kld_source_null` is read as a *measurement* on this architecture rather than as a
number the mechanism drives to zero. The cancellation "by construction" the mechanism was built for
would need the delta defined as $D(a) - D(a^\varnothing)$ — a posterior change, which this
architecture does not make; §14 records it as an open limitation with its trigger.

**One coupling between §8.1 and §8.2, recorded because it is real.** `_prior_clock` is an encode
through the K/V path, so its liveliness is a function of `lag_kv_source`: whole-prefix under
`encoder` and under this cell's $387$-step stem, bounded by the stem's reach on the conv-Transformer
cell, and **exactly constant** under `adapter`, where the pathway has no memory and the clock
collapses back to the staircase that is provably inert. The two keys are not independent, and an arm
choosing `adapter` is also choosing to make this mechanism dead.

## 9. Forward return dict

`SeqVaeLagAttnCfs.forward(y_st, y_ph, u_stream, anchor_phase=None, anchor_stride=None)` returns
**twenty-three keys** at the shipped configuration: the family's twenty, the two the anchor axis
needs, and `persistence` — which is present only under `persistence_residual`, so a model with that
key off returns twenty-two and is bitwise the contract that predates the mechanism.

- `mu_base`, `logvar_base`, `mu_full`, `logvar_full` — $(B, A_{\max}, H, C_{\mathrm{keep}})$, so
  $(B, 5, 30, 98)$ at the shipped training geometry, $(B, 136, 30, 98)$ at the dense one, and
  $(B, \cdot, 15, 102)$ ungated.
- **`anchor_index`** $(B, A_{\max})$ `long` and **`anchor_valid`** $(B, A_{\max})$ `bool` — the
  decoded anchors and which of them are real. The dtypes are part of the contract: the first gathers
  and scatters, and the second multiplies into a float mask, where a float that is silently truthy
  would defeat the padding.
- `mu_prior`, `logvar_prior`, `raw_logvar_prior`, `mu_post`, `logvar_post`, `z_prior`, `z_post` —
  $(B, 300, 64)$, unchanged.
- `target_state`, `source_state` — $(B, 300, 128)$; `attended_source_heads` $(B, 300, 4, 32)$;
  `attn_weights` $(B, 300, 4, 91)$ — shapes unchanged. **`source_state` is now the lag attention's
  K/V tensor** rather than the deep source state by definition (§8.1): under `encoder` those are the
  same object, and under a local arm it is the stem's or the adapter's output. It is the thing the
  model reads, which is what makes both controls' re-pointing a one-line consequence rather than a
  second decision.
- `kld_per_t`, `kld_per_t_per_head`, `source_kl_lag_map`, `mu_prior_sat_frac`, `delta_mu_sat_frac` —
  unchanged.
- **`persistence`** $(B, A_{\max}, C_{\mathrm{keep}})$ — the anchor's own target vector, under
  `persistence_residual` alone. It is returned rather than re-gathered because a control or an
  estimator that re-decodes must carry the *same* tensor the forward decoded with (§5), and it is
  indexed by anchor **position** rather than by sequence step, which is the contract a caller
  narrowing it to a subset of anchors has to honour.

**No `decoder_state` and no `delta_mu_src`**: the decoder receives the latent and the target-only
persistence vector, and nothing else, so there is no source bypass to report.

The anchor set is built inside `forward` because the four forecast tensors and the target must be
gathered at the same anchors and a second computation could disagree; returning it makes it part of
the forward contract, testable and available to the diagnostic page without a second code path.

## 10. What is measured

Eleven readouts this package adds — ten on both stages and one on the evaluation stages alone.
Almost all are partial sums or fractions of quantities the objective already computes, so they add no
second definition of anything; `kld_source_null` is the exception and is the only one that costs a
forward pass.

| Metric | Stages | What it separates |
| --- | --- | --- |
| `target_warm_frac` | train, val | the budget-and-floor pairing; a constant, resolved at construction, **exactly $1.0$** |
| `anchors_per_sample` | train, val | the tiling actually firing; $[10, 11]$ in train, $51$ in val at the shipped forecast clock |
| `source_lag_warmth_frac_st` | train, val | attention mass landing on lags where the first stored source block is warm |
| `source_lag_warmth_frac_ph` | train, val | the same for the second, which is the block with the problem |
| `pred_gap_warm_lo` / `_mid` / `_hi` | train, val | whether slow channels forecast differently from fast ones |
| `pred_gap_novel_lo` / `_mid` / `_hi` | train, val | how much of the score is a forecast rather than an inversion of history |
| `kld_source_null` | val | the KL floor the availability clock induces with no source content |

**Two of them are guards rather than results.** `target_warm_frac` must read exactly $1.0$ and
`anchors_per_sample` must sit at its geometry-derived value; a row outside either means the geometry
broke, not that the model learned something. `target_warm_frac` is computed **once at construction**
and emitted as a constant column, because given the constructor's pairing refusal and the anchor
range it is identically $1.0$ — recomputing a would-be four-dimensional density every step would be a
tautology evaluated per batch. What the column is for is **provenance**: a value other than $1.0$ on
a logged row means the checkpoint was built by code predating that refusal.

**The three warm-up tertile columns partition the $98$ kept channels by $W'$ and recompose to
`pred_gap`** over the same denominator, exactly as the block split does. They are not a restatement
of `pred_gap_st` / `pred_gap_ph`: the kept set is $32$ channels of the first stored block plus all
$66$ of the second and both span nearly the same rebased range, so tertiles by $W'$ cut *across* the
block boundary. The partition is by **rank** of $W'_c$ rather than by its value, so the boundaries
move when the budget moves rather than sitting at step counts a rebuilt dataset would invalidate.

**The three novelty tertile columns cut the same axis a third way, and answer the question the
pooled block score cannot.** $\nu_c$ is the share of a target coefficient drawn from raw samples the
anchor has not seen — a property of the bank, computed at write time and stored per block as
`causal_novelty_frac`. On the shipped kept set it runs from $1.000$ on the fastest channels to
$0.026$ on filter $30$, the slowest one the budget keeps: over the whole $120$ s horizon, $97.4\%$ of
that coefficient is determined by fetal heart rate the anchor has already observed. `pred_gap` sums
$H \cdot C_{\mathrm{keep}} = 2940$ coefficients without distinguishing the two, so a good score on
the low-$\nu$ end is the model inverting its own differently-delayed history and on the high-$\nu$
end it is a forecast. Both are worth having and they are not the same claim, which is why the split
ships rather than the pooled number alone.

This is **not** the warm-up split renamed. They descend from one filter ladder, so the two
assignments correlate — but the warm-up asks when a channel became a function of the recording at
all, and the novelty asks how much of what it reports at a *scored* step is still ahead of the
anchor, and the slowest kept channel is warm across the whole window while carrying $\nu = 0.026$.
The vector travels as the constructor keyword `target_novelty_frac`, in **declared** channel
coordinates and gathered through the keep-index exactly as `target_channel_weight` is, so the
ungated comparison arm — built by removing every resolved channel tuple and keeping the readouts —
still receives a vector of the right width. A *gated* run cannot fall back: `warmup_model_kwargs`
refuses a feature-target model whose shards carry no novelty vector rather than defaulting one,
because a zero would report every channel as pure history and a one as pure forecast.

All four gap readouts inherited from `lag_attn_fs` are kept, with `TARGET_BLOCK_SPLIT` re-pointed
from $43$ to $36$. `SOURCE_BLOCK_SPLIT = 36` is the boundary the two source-warmth columns are
reported either side of, and the split is not decoration: the two blocks' rebased warm-ups are
$0 \ldots 278$ and $41 \ldots 134$ declared, and $55 \ldots 92$ and $71 \ldots 92$ once the
alignment's shift is added to the survivors, so a pooled figure would let the first block's fast
channels carry the fraction while the second's are the last to arrive.

**Both columns are computed from $W'_c + d_c$**, the same quantity the availability mask and the
anchor floor use — see §8 for why the earlier $W'_c$-only version made
`source_lag_warmth_frac_st` a constant $1.0000$. One consequence is worth stating rather than
discovering: `_resolve_block_warm_steps` treats an **empty** block as warm at every step, on the
principle that a zero in the CSV reads as a measurement rather than as an absence. No shipped
arm of *this* cell has an empty source block — `up_st` keeps $30$ and `up_ph` $9$ — but the
raw-target cells' lower reference does empty one, so either fraction must be read beside its
block's kept width and never alone. The shipped source reference (§3.1) narrows `up_ph` from $15$ to
$9$, which is the largest move either width has made, and it is exactly the trade that reference was
priced against.

**All ten net-side readouts are emitted from `_resolved_forecast_gaps`**, which is this family's
per-package metric hook — the parent's `compute_loss` merges whatever it returns — and is the only
hook available, because the anchor seam lives in the shared objective rather than in a subclass.
`kld_source_null` is task-side, like `shuffle_penalty`, because it needs a second forward dict the net
never sees.

**One inherited readout is re-pointed rather than left alone.** `_mu_gap_rms` rebuilds
`forecast_mask` and `kl_mask` itself with no anchors, and its own docstring states that it uses "the
KL's own anchor support … so the two cannot drift". Under tiling that stated invariant *fails*:
`mu_post_prior_gap_rms` would average the latent belief shift over all $136$ anchors while the
`source_conditioned_kl_raw` printed beside it averages over $\approx 4.6$, and the two are read
against each other. The override takes the same anchor set, which restores the property the function
already claims. Leaving it dense and renaming it was the alternative; consistency is cheaper than a
second name and a second explanation.

`LagAttnCfsTrainer.TRACKED_METRICS` carries **99** entries: the shared surface, the two-sided
sibling's four gap columns on both stages, this package's ten on both stages, and the source-null
KL on validation alone. A `train/kld_source_null` is deliberately absent — it is a readout that never
enters the objective, so the column would be NaN in every row of every run.

## 11. What is drawn

Three per-run figures and one offline, and the division is not cosmetic: a figure that is a constant
of the *shard* rather than of the run would be an identical file written into every run directory.

1. **The warm-up staircase**, inside the `input_target` / `input_source` page rows. A warm-up
   boundary is a per-channel step function and the shipped row already draws one with `ax.step(...)`;
   filling it with $W'$ draws it with no new artist. The row builder is replaced rather than
   inherited, because the shipped one calls `describe_streams` — welded to the production two-sided
   Morlet bank, which did not produce these coefficients and refuses these widths — **inside a
   handler that warns and continues**. Leaving it in place costs two page rows and one log line with
   a green suite, which is why the test asserts the *absence of the warning* rather than the presence
   of the rows.
2. **The run-level warm-up budget figure**, `causal_warmup_budget`, reusing `_budget_panel` by object
   identity. Feeding $\delta = W'$ and $\rho = \Delta W'$ turns each bar into $[-\Delta W'_c, 0]$: a
   backward settling length ending at the anchor's causal endpoint. The reading is the mirror image
   of the two-sided figure's — there a bar that ends after $0$ reaches into the window it is meant to
   forecast; here a bar that *starts* before $0$ is how long the channel spent becoming honest. It is
   drawn from the resolved budget rather than from the network, because its subject is the channels
   the budget **dropped**, and a dropped channel's $W'_c$ is exactly what the checkpoint does not
   carry.
3. **The anchor overlay** on the forecast row: the floor, the decoded anchors read off `anchor_index`,
   and their windows — read from the forward output, so the figure cannot disagree with the loss. The
   page is produced at the dense set and phase $0$, which is *not* the geometry a training step used,
   and the row states the training stride beside it.
4. **The budget tradeoff curve**, `causal_warmup_tradeoff`, produced offline by `causal_warmup.py`'s
   own entry point: kept channels, admitted anchors and tiles against every candidate budget, with
   the shipped one marked. Its anchors are computed from the **survivors'** own maximum rather than
   from the threshold — a threshold of $151$ keeps the identical channels as $134$ and would
   otherwise read $17$ steps worse.

### 11.1 The page's fifteen rows

The shared builder reserves seven; `input_stream_panels` adds two; `forecast_extra_rows` — the
seam resolved off the task beside the other three — adds the six of `CAUSAL_EXTRA_ROWS`, directly
below the forecast row so everything the model *produced* stays contiguous and the input rows still
sit against the latent they feed.

| # | row | what it is |
|---|-----|------------|
| 1 | `raw` | the raw FHR and UP traces, for physiological context |
| 2 | `forecast` | three channels by $2\sigma$ calibration as offset lanes, plus the anchor overlay and one anchor's error map |
| 3–5 | `pred_truth`, `pred_base`, `pred_full` | $Y^{+}$, $\mu^p$ and $\mu^q$ over all $98$ kept channels, one colour scale |
| 6 | `pred_skill` | $\lvert Y^{+}-\mu^p\rvert - \lvert Y^{+}-\mu^q\rvert$ — `pred_gap` resolved per channel and per step |
| 7 | `pred_sigma` | $\sigma^q$, beside the error it predicts |
| 8 | `pred_gap` | each drawn window's block score, base against full, plus two per-channel profiles |
| 9–10 | `input_target`, `input_source` | the streams as the encoders receive them |
| 11–15 | `latent`, `kld_dims`, `kld_total`, `lag_attn`, `kl_lag_map` | the shared builder's, unedited |

Three properties of that block are load-bearing.

**Rows 3–8 draw one tiling, not six.** The lane row resolves `_tiling_anchors` once and hands the
result down as a `_Stitched`. Six rows each walking the decoded anchor set would be six pictures
that only look aligned, and the misalignment has no shape error in it — the same failure the
`forecast_rows` seam exists to prevent, one layer up.

**Row 8's two curves are the objective's own number.** They go through `raw_sample_score` under
`forecast_mask`, the functions `_resolved_forecast_gaps` itself reduces, so a window's height is in
the same nats as the `nll_base_block` and `nll_full_block` in the page's title and the shaded area
between the curves is `pred_gap` restricted to that window. This is the one place the page computes
rather than lays out, and it is deliberate: an error curve drawn from a re-derived formula is the
one diagnostic that can disagree with the run it is diagnosing. It needs two values the arrays do
not carry — the likelihood and the coverage floor — so the task binds them into the seam from where
the objective takes them. A batch with no `weight` cannot have the mask built, so the row is
annotated rather than raised over.

**Every channel axis reads top-down**, through `top_down_extent`: coefficient $0$ at the top, the
scattering block above the phase block, on the input rows, the error map and all five field rows
alike. The convention is the shared page's, so the three sibling models moved with it. A reader
carries a channel index down the page and it is in the same place on every row.

**Every inset sits in the warm-up prefix**, through `_prefix_boxes` — row 2's per-anchor error map
and row 8's two per-channel profiles alike. The two-sided sibling puts its error map in the *right*
margin, and that is right for its tiling and wrong for this one: both pages leave one span of the
forecast row undrawn, but its tiling stops short of the recording's end and this one runs to it, so
the blank corner is the tail there and the prefix below the anchor floor here. The prefix is blank
by construction rather than by luck — no anchor exists below $F$, and $F \ge B - 1$ is refused at
construction rather than merely expected — and at the shipped geometry it is $134$ of $300$ steps.
Inheriting `_ERROR_MAP_BOX` unedited put the panel over the last windows of the very forecast it is
a detail of, so `_draw_error_map` takes the box as an argument and each page passes its own.

## 12. DDP, and the branches that are construction-time decisions

Production runs under plain `"ddp"` with `find_unused_parameters=False` under `gaussian_nll`, so
every parameter must be reachable; `mse` starves the decoder's log-variance heads and selects
`ddp_find_unused_parameters_true`. Both are measured, on the guarded *and* the ungated arm — the
guarded one is the only configuration in which the adapters carry an availability projection at all.

A parameter multiplied by an identically-zero tensor **is** reachable. What breaks
`find_unused_parameters=False` is a parameter left out of the graph by a Python-level branch on a
tensor value, so the availability terms are added unconditionally in the forward and every branch is
a construction-time decision on a module's existence. The per-segment phase is derived per rank from
that rank's own samples and introduces no collective; $A_{\max}$ is a geometry constant, so no rank
can disagree on shape; and neither the stride nor the phase is read off `self.training`, so no branch
depends on the dropout switch.

**Every new parameter is reachable in the graph.** The clock projection is added unconditionally in
the prior head's forward when the flag is on; the persistence weight multiplies a tensor the decoder
always receives when it is built; the lag bias exists whether or not its seed is flat. After one
backward on a shipped-configuration model no trainable parameter has `grad is None`, which is what
`find_unused_parameters=False` requires and what §17's off-states must not quietly break. The
horizon weight is a **non-persistent buffer** and therefore absent from the state dict, which is
deliberate: a checkpoint stays loadable across horizons.

**A run's own artifacts state every independently-toggleable mechanism.** Each is its own
configuration key and each is a leaf of `model_config.VAE_model`, so all of them land verbatim in
the resolved-config artifact the run writes beside its checkpoints, and the architecture ones land
again in the checkpoint's own `model_kwargs`. `causal_leg_alignment` names which phase-harmonic
operator built the phase blocks and is compared against the shards' own root attribute;
`causal_align_reference` and `causal_align_reference_source` name the two clocks and resolve to
$\tau^y_{\mathrm{ref}}$ and $\tau^u_{\mathrm{ref}}$ from the data; `lag_kv_source`,
`prior_availability_input`, `persistence_residual`, `horizon_weight_halflife_steps` and
`alibi_slope_scale` name the five architecture switches. The resolved *consequences* land in the
startup log's budget summary — both references in seconds, the inter-stream offset, the shift range
and the surviving counts per block, per stream — and the evaluation console prints the configured
arm label, both resolved clocks, the offset and `lag_kv_source` on the same block as the delay, with
the same three readings in `summary.json`. The logged reference is $\tau_{\mathrm{ref}}$ as
`causal_delay_s` reports it while the logged shift range is already scaled by $\kappa$ (§3), so
`shift 0-85 steps` against `402.1604 s` is the consistent pairing on the target stream and
`shift 0-97` would mean a resolver predating the factor. No tracked metric is added for any of them:
they are constants of the configuration, so a per-step column would be the same value in every row.

**That disclosure is a guard rather than a convenience, and it exists because the failure it catches
has happened.** The driver builds a run's kwargs by sweeping `inspect.signature(MODEL_CLS.__init__)`
and **silently drops** any config key the class does not re-list, so an arm can train as the
baseline with no error and no metric saying so. Printing the *configured* value beside the *built*
one is what makes that visible; the sweep-arm configs additionally put the arm in `run_name` and
`tags.variant`, so the run's own directory name carries it.

**Two evaluations of one checkpoint on one shard must produce an identical `metrics_history.csv` row
set.** Identical anchor indices are necessary and not sufficient, since the reparameterisation
$\epsilon$ and the permutation generator also move. `check_run.py` compares two run directories'
rows as text for exactly that reason.

## 13. Parameter budget

Measured on constructed models at the shipped warm-up budget, not predicted. `tests/test_docs.py`
re-measures every total below by constructing the models rather than comparing against literals, so a
legitimate change to a shared imported component re-costs this table instead of failing an unrelated
assertion.

**Two rows per cell, and both are the record.** The **shipped** row is the revised default: local
K/V, the prior clock, the persistence residual, the weighted horizon axis, the flat lag-bias seed
and the two clocks of §3.1. The **off-state** row is every switch of §17 at its inert value with the
single shared clock restored, which is bitwise the model that shipped before this revision — and it
is the row on which the two-sided comparisons below are still readable, because those cells never
took the new keys.

| | conv-LSTM encoders | conv-Transformer encoders |
| --- | ---: | ---: |
| **causal feature, shipped** ($C_{\mathrm{keep}} = 98$, $c_u^{\mathrm{keep}} = 39$) | $\mathbf{4{,}655{,}987}$ | $4{,}284{,}556$ |
| causal feature, shipped but one shared clock ($c_u^{\mathrm{keep}} = 47$) | $4{,}658{,}035$ | $4{,}286{,}604$ |
| causal feature, shipped but unaligned ($c_u^{\mathrm{keep}} = 51$) | $4{,}658{,}803$ | $4{,}287{,}372$ |
| causal feature, shipped but ungated ($C = 102$, $c_u = 51$) | $4{,}642{,}419$ | $4{,}270{,}988$ |
| causal feature, `lag_kv_source: encoder` at the shipped clocks | $5{,}163{,}866$ | $5{,}072{,}524$ |
| causal feature, `lag_kv_source: adapter` at the shipped clocks | $3{,}851{,}635$ | $4{,}183{,}564$ |
| **off-state, guarded and aligned** ($c_u^{\mathrm{keep}} = 47$) | $\mathbf{5{,}146{,}334}$ | $5{,}054{,}992$ |
| off-state, guarded and unaligned ($c_u^{\mathrm{keep}} = 51$) | $5{,}147{,}102$ | $5{,}055{,}760$ |
| off-state, ungated ($C = 102$) | $5{,}130{,}598$ | $5{,}039{,}256$ |
| two-sided feature, guarded ($C_{\mathrm{keep}} = 78$) | $5{,}126{,}326$ | $5{,}034{,}984$ |

**Ungated means the whole guard**, the warm-up mask and both common clocks together, and that is
forced rather than chosen: a shift vector is positional over the *survivors*, so a stream that has
lost its keep-index has no width for one to be positional against and `ChannelDelay` refuses it by
name.

### What the revision costs, factorised

**At fixed clocks the five architecture switches cost $-488{,}299$ on this cell** and
$-768{,}388$ on the conv-Transformer one, and each term is a module rather than a rounding:

| Term | conv-LSTM | conv-Transformer | What it is |
| --- | ---: | ---: | --- |
| the deep source encoder, not built | $-1{,}312{,}231$ | $-888{,}960$ | `lag_kv_source` is not `encoder`, and nothing else consumed it |
| the local K/V stem, built | $+804{,}352$ | $+100{,}992$ | the parent encoder's own convolution blocks plus an output norm |
| the prior's clock | $+16{,}640$ | $+16{,}640$ | $2 \times 128$ for its own `LayerNorm` and $128 \times 128$ for the bias-free projection |
| the persistence weight | $+2{,}940$ | $+2{,}940$ | $H \times C_{\mathrm{keep}} = 30 \times 98$, one raw parameter |
| the horizon weight | $0$ | $0$ | a non-persistent buffer, so not a parameter and not a state-dict key |
| the flat lag-bias seed | $0$ | $0$ | the same $(\text{num heads}, L)$ parameter, seeded differently |

$-1{,}312{,}231 + 804{,}352 + 16{,}640 + 2{,}940 = -488{,}299$, and
$-888{,}960 + 100{,}992 + 16{,}640 + 2{,}940 = -768{,}388$. A delta that does not decompose into
exactly these means something else moved.

**The two stems are very differently sized, and that is a schedule rather than a design difference.**
Each reuses *its own* parent encoder's convolution schedule, so the two arms differ in what is
removed rather than in two independently chosen front ends. Here that schedule is
$(3, 5, 11, 15, 15)$ at dilations $(1, 2, 4, 8, 16)$, reaching
$1 + \sum_b (k_b - 1) r_b = 387$ steps; on the conv-Transformer cell it is $(5, 9)$ at $(1, 2)$,
reaching $21$. **387 steps is longer than the $91$-step lag window and longer than the $300$-step
sequence**, so on *this* cell `conv_stem` bounds the recurrence and not the memory: a lag profile
read off this arm must be read beside that number, and the K/V localisation argument of §8.1 is only
partially in force here. The conv-Transformer cell is where that argument gets an honest test.

**The dual reference costs $-2{,}048$ in both cells**, and it factorises exactly: the source adapter
loses eight channels from two $d_{\mathrm{model}}$-wide linears, its input linear and its
availability projection, at $-8 \times 128 \times 2 = -2{,}048$. The target stream contributes
nothing, which is the whole point of the second key.

**The alignment costs $-768$, in every one of the four causal cells**, and it factorises exactly:
the source adapter loses four channels from two $d_{\mathrm{model}}$-wide linears at
$-4 \times 128 \times 2 = -1{,}024$; and both adapters gain a start-of-record vector (§7) at
$+2 \times 128 = +256$. The unaligned and single-clock rows are kept as named comparison arms rather
than deleted, because they are what `causal_align_reference: null` and
`causal_align_reference_source: null` still build and what the pre-registered comparison is read
against.

### The guard, at the shipped configuration

**The guard costs parameters here and saves them on the two-sided sibling, and both numbers are
right.** Guarded minus ungated is $+13{,}568$ on the shipped model against $-9{,}662$ on
`lag_attn_fs`, because the two guards drop very different numbers of channels. Here the budget drops
$4$ target channels of $102$ and the alignment $12$ source channels of $51$, so the machinery the
guard adds still dominates the narrowing it buys:

$$\underbrace{128 \times 98}_{\text{target availability}}
+ \underbrace{128 \times 39}_{\text{source availability}}
+ \underbrace{2 \times 128}_{\text{start embeddings}}
- \underbrace{514 \times 4}_{\text{decoder head}}
- \underbrace{128 \times 4}_{\text{target input linear}}
- \underbrace{128 \times 12}_{\text{source input linear}}
- \underbrace{30 \times 4}_{\text{persistence weight}} \;=\; +13{,}568 .$$

There the reach budget drops $31$ of $109$, so the narrowing dominates instead. Stated here so the
next reader does not treat two correct numbers as a contradiction.

**The persistence weight is the seventh term and it is new**, because $w$ is
$(H, C_{\mathrm{keep}})$: the guard narrows the decoder's channel axis, and the residual's weight is
positional against exactly that axis. Off, the term is absent and the sum is the six-term one — which
is what the off-state row measures, at a source stream of $47$ channels rather than $39$:

$$\underbrace{128 \times 98}_{\text{target availability}}
+ \underbrace{128 \times 47}_{\text{source availability}}
+ \underbrace{2 \times 128}_{\text{start embeddings}}
- \underbrace{514 \times 4}_{\text{decoder head}}
- \underbrace{128 \times 4}_{\text{target input linear}}
- \underbrace{128 \times 4}_{\text{source input linear}} \;=\; +15{,}736 .$$

The two identities differ in exactly three places, and each is one of this revision's own decisions:
the source availability projection is $39$ wide rather than $47$ and its input linear loses twelve
channels rather than four (§3.1's second clock), and the persistence weight is there at all (§5).

### The two axes, read on the off-state row

**The encoder axis: $91{,}342$** at the off-state, guarded against guarded — the conv-Transformer
encoders are that much smaller at a fixed target. It is **identical** to the reduction the same two
encoders buy in the two-sided feature domain and in the raw domain at the same budget, which is what
a difference living entirely in the two history encoders must look like, and it is identical again
on the ungated arm.

**At the shipped configuration the same axis is $371{,}431$**, and it is identical across the
shipped, single-clock, unaligned and ungated rows — so it is still the two history stacks alone and
nothing else. It moved because *both* history stacks moved: the target encoders differ by
$-331{,}929$ as they always did, and the two stems by $+703{,}360$ where the two deep source
encoders differed by $+423{,}271$. $-331{,}929 + 703{,}360 = 371{,}431$. The comparison against the
two-sided pair's $91{,}342$ is therefore only meaningful on the off-state row, where both pairs
carry the same source module; that is stated rather than left for a reader to trip over.

**The target axis: $+20{,}008$** against `lag_attn_fs`, guarded against guarded on the **off-state**
row, and identical in both encoder families. It decomposes into exactly **two** terms, each measured
parameter by parameter:

| Term | Value | What it is |
| --- | ---: | --- |
| the decoder's output head | $+10{,}280 = 514 \times (98 - 78)$ | two per-channel output rows plus their biases, at $d_{\mathrm{hidden}} + 1 = 257$ each |
| the two input adapters | $+9{,}728$ | $128 \times (98 - 78)$ and $128 \times (47 - 29)$ on the input linear *and* the availability projection |

$10{,}280 + 9{,}728 = 20{,}008$. `horizon_depth` is **not** a term — it stays at the sibling's $4$ —
so a delta that does not decompose into exactly these two means something else moved. **It is read
on the off-state row deliberately:** `lag_attn_fs` is a two-sided cell and never takes the new keys,
so at the shipped configuration the difference between the two models is dominated by mechanisms one
of them does not have, and no target-axis reading survives it.

**The start-embedding term used to be the adapters' third and is now exactly zero.** It contributed
$-256$ while this cell built neither vector and `lag_attn_fs` built both; the alignment builds both
here too (§7), so the two models agree and the term vanishes — which, together with the source
stream narrowing to $47$ on that row, is why the target-axis delta fell by precisely $768$.

**The horizon embedding used to be the third term and is now exactly zero.** It contributed
$-3{,}840 = -15 \times 256$ while this cell forecast one minute against `lag_attn_fs`'s two; both
now forecast $30$ steps, so the two embeddings are the same size and the term vanishes — which is
why the target-axis delta grew by precisely $3{,}840$.

Both causal off-state rows gained exactly $+3{,}840 = 15 \times 256$ when the horizon moved to $30$,
and that is the **whole** parameter cost of doubling the forecast: the horizon embedding is the only
tensor in the model whose shape carries $H$ — the persistence weight of §5 is the second, and it did
not exist then. The projection MLP, the dilated refine stack, the FiLM generators, the
horizon-attention blocks and the two output heads are all shared across horizon tokens. The
two-sided row did not move, because it was already at $30$.

## 14. Deliberate limitations

- **The nats are budget-local, model-local and horizon-local.** §5. Recorded, not fixed.
- **`_mu_gap_rms` is overridden onto the tiled anchor set.** §10. The alternative — leaving it dense
  and renaming the column — would put two differently-denominated numbers on one page under two
  names, and a reader comparing `mu_post_prior_gap_rms` against `source_conditioned_kl_raw` would be
  comparing an average over $136$ anchors against one over $\approx 4.5$.
- **Group-delay compensation is per channel, not per channel pair.** The second `lean-limit:` note
  below. `causal_delay_s` is now read — by the warm-up resolver, which turns it into the per-channel
  shift §3 describes — so each stream's entries describe one instant; what is still uncorrected is
  the *pair*-indexed bias between a source channel's clock and a target channel's.
- **Every local measurement is in-sample.** `dataset_kwargs` is shared between the two loaders and
  cannot carry a per-split GUID filter, so a dev-box run validates the objective's optimisation
  behaviour and says nothing about generalisation.
- **No mixed precision** (`precision: "32-true"`) and **`compile: false`**, the latter permanently:
  the `nn.LSTM` encoders, the checkpointed attention region and the data-dependent boolean mask
  indexing behind `kld_active_frac` each break TorchInductor independently.
- **No warm start.** `core_model_checkpoint` stays `null`: a blob from any sibling carries a different
  `model_class` stamp and a decoder head of a different width, and the guard refuses it — correctly.
- **Checkpoint compatibility with the pre-revision model is deliberately broken.** The constructor
  and state-dict changes of §17 mean a pre-revision blob does not load into a shipped-configuration
  model. `load_checkpoint_strict` refuses rather than partially loads and `check_model_class` still
  guards the class, so the failure is by name; no migration shim is built, because the off-state
  arm exists for exactly the case where the old weights are wanted.
- **The K/V receptive field is the lag resolution floor, and on this cell it is not local.** §8.1
  and §13. The stem reuses this parent's own convolution schedule, which reaches $387$ steps —
  longer than the lag window and longer than the sequence — so on this cell `conv_stem` removes
  unbounded recurrence rather than whole-prefix memory. Giving the stem a schedule of its own
  (kernels $(3, 5, 11)$ at dilations $(1, 2, 4)$ reach $51$ steps; the first two blocks alone reach
  $11$) would be a new decision rather than a reuse, and it is not taken here.

> lean-limit: the prior's clock cancels no part of the KL's **mean** term, because the posterior is
> a bounded residual on the prior and that term is a function of the delta head alone; replace with
> a delta defined as $D(a) - D(a^\varnothing)$ -- and `posterior_logvar_mode` back to `residual` for
> the variance half -- when the owner accepts a change to the posterior parameterisation the whole
> coupling readout is defined on.

§8.2 carries the decomposition. The mechanism is kept at `prior_availability_input: true` rather
than removed, and it is not dead: its off-state is bitwise the pre-revision prior, the clock it
supplies is the object any posterior-side fix would also need, and it can still reach the variance
half of the divergence. What it cannot reach is the half the acceptance criterion was written
against, and that is a property of the parameterisation rather than of the clock.

> lean-limit: the persistence residual ships for the feature-target cells only; replace with an
> anchored raw persistence input when the raw-target cells' fast-step NLL shows the same
> suppression signature on a trained revised run.

The finding the residual answers — that the channels the model abandons are the fast ones, whose
persistence $R^2$ is negative from horizon step $4$ — is a **feature-domain** measurement, and a raw
persistence input is a different object: the raw grid has $16$ samples per decimated step and no
channel axis to be positional against. The two raw-target cells therefore do not take the key at
all, and their constructors refuse it by name rather than building a weight no forward reaches.

> lean-limit: the causal warm-up is paid once per $22$-minute segment rather than once per recording,
> costing roughly half the available forecast supervision; replace with a left-context rebuild when a
> measured run shows the anchor floor rather than the source pathway is the binding constraint on
> `pred_gap`.

The transform runs per segment and the chain is batch-invariant, so the one-sided filters restart from
assumed history once per segment. Prepending left-context before the transform would drive every $W'$
to zero: roughly $4{,}688$ samples ($19.5$ min) recovers the $36$ kept scattering channels and
roughly $10{,}176$ ($42.4$ min) additionally recovers the seven dropped ones, returning $c_y$ to
$109$ and $c_u$ to $58$ (before the alignment's own four) and making the causal-versus-two-sided
comparison exact. It costs roughly
$1.9\times$ to $2.9\times$ build compute, loses the first one to three segments per GUID, and means a
segment's coefficients are no longer reproducible from its own stored `fhr`/`up`. **The model side
needs no change against such a dataset**: the budget resolves to every channel and the floor to the
model's own $30$.

> lean-limit: the lag map is an attribution over stored-coefficient time whose bias is now the
> single constant $\kappa(\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}})$ plus the one-sided
> intra-band residual, rather than a $1167$ s pair-dependent range; replace with a physical-lag
> readout when a nonzero target reference can be divided out. On the *stored* forecast clock it
> cannot -- the scored target is a stored coefficient on each channel's own clock -- and under
> the shipped **physical** forecast clock (see the amendment at the end of this document) it
> can: every scored element describes one physical instant, the reference to divide out is the
> forecast clock's own $\tau_{\min}$, and the offset the identity subtracts becomes
> $\tau^u_{\mathrm{ref}} - \tau_{\min}$.

One-sidedness and zero latency are different properties and this family buys only the first. A
causal channel's raw composed group delay $\tau_c$ spans $13.3$ to $791.0$ s on `fhr_st`, $20.5$ to
$402.2$ s on `fhr_ph` and $150.8$ to $402.2$ s on `up_ph`; the alignment (§3) re-indexes every kept
target channel onto $\tau^y_{\mathrm{ref}} = 402.1604$ s as reported — a common *realised* delay of
$\kappa\tau^y_{\mathrm{ref}} = 351.9$ s — and every kept source channel onto
$\tau^u_{\mathrm{ref}} = 288.2672$ s, realised $252.2$ s, so what remains inside a stream is the
intra-band residual and the $\le 1.99$ s quantisation residual rather than a thirteen-minute spread.

**The intra-band residual is a one-sided bias, not a symmetric spread, and §3's $\kappa$ is the
correction for its systematic part.** $\tau_g(\nu)$ falls off the centre frequency as
$b^2/(b^2 + (\nu-\xi)^2)$, so it is *maximal* at $\xi$ and a channel's own passband can only pull
the realised delay **below** the number used to align it — at the $-3$ dB edges, where
$\nu - \xi = 0.435\,b$, down to $0.841\,\tau_g(\xi)$. Describing that as "$\pm 16\%$" is wrong in
direction: there is no $+16\%$ side. Aligning on $\kappa\tau_g$ rather than on $\tau_g$ removes the
centre of that downward pull, and the measured residual after it is a few per cent — median
realised/reported $0.9032$ against $0.875$ applied, and $0.882$ over the nine channels the $4$ s
grid barely quantises. What is left is a delay *profile* per channel that no per-channel scalar can
carry; a minimum-phase spectral factorisation of the Morlet magnitude would supply one, and is
declined for the same reason the bank declines it.

**The model side and the transform side are on different delay conventions, on purpose.** Only the
model-side alignment carries $\kappa$: `channel_alignment_delays` / `_align_stream` re-index what
the shards already hold, so the factor costs nothing but a re-resolve. The transform-side
`causal_scattering.leg_alignment_shift`, which puts the two legs of a phase-harmonic pair on one
clock, still uses $\tau_g$ unscaled — applying $\kappa$ there would change stored coefficient
values and invalidate every existing causal shard, so it waits for a rebuild that is being paid for
anyway. `causal_delay_s` likewise still ships $\tau_g$ and should: it is the right number to
*report* a channel's staleness with, and the consumer that has to *cancel* a delay rather than
quote one is the one that applies the factor. This is a stated limitation rather than an oversight,
and until a rebuild lands the two numbers must not be read as the same quantity.

The **forecast** claim is exact under either — a coefficient at $t$ is a function of $\{x(s) : s \le t\}$, so forecasting
$t + 1 + \tau$ from history up to $t$ is a genuine forecast whatever the internal latency, and
`pred_gap` needs no correction. Any lag or transfer-entropy-style reading does:

$$\mathrm{lag}_{\mathrm{physical}} = \Delta\ell \;+\; \kappa\big(\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}\big) \;+\; \mathrm{shift}_{\mathrm{preprocessing}},$$

with $\Delta = 4$ s and $\mathrm{shift}_{\mathrm{preprocessing}} = -20$ s from the UP advance applied
before any transform. The alignment collapsed the two $\tau$ terms from channel-dependent quantities
of the same order as the lag search itself — the spread of $\tau^u$ across `up_ph` alone was $251$ s
against a $364$ s lag window — to one reference each. Under §3.1's dual scheme the bracket is a
single **known non-zero constant**, $\kappa \cdot (-113.8932) = -99.65$ s, carried explicitly
through the preflight causality record and printed on the evaluation console beside the delay — and
that is what puts a $20$–$60$ s physiological delay at lags $8.5$–$47.5$ of $[0, 90]$ instead of
below the near edge. What still does not follow is a lag between *signals*: a **target** reference
of zero is what that would need, and this cell's target is itself a stored coefficient on its own
clock. So this cell's lag map remains an attribution over stored-coefficient time with a known
offset, and every lag-resolved caption still says so. The two raw-target cells are where the target
reference is zero and the constant is fully recoverable.

> lean-limit: `lag_floor` is configuration for a value no shipped arm varies; replace with a
> constructor-level constant, or with a resolved source floor, when a run's
> `source_lag_warmth_frac_ph` falls below the fraction of searched lags that are warm at all.

It ships at $0$, where the lag mask is bitwise the sibling's, and it is kept as a knob by direction
rather than by evidence: the argument for deleting it — configuration for a value that never varies,
and a defaulted keyword added later strands no checkpoint — is a good one. Its cost is held to one
bitwise-equality test.

### What the evaluation closed, and what it deliberately did not

There is now an evaluation pipeline for this cell: `eval/`, a `ModelBinding`, twenty-one registered
analyses over one shared collection pass, two durable per-recording tables, ten pre-registered
acceptance verdicts, bootstrap intervals over recordings, and an offline gate that reads
`summary.json` and imports no `torch`. `eval/EVAL.md` is its contract and `eval/FIGURE_GUIDE.md`
documents every figure; both are bound to the code by test. It is a **fork** of
`teb_vae/lag_attn_rws/eval` edited for this target domain, and it carries four named measures
against the drift a fork invites — the model-free primitives stay shared, a sibling-agreement test
re-derives the shared arithmetic through both packages, a machine-checked `divergences.json`
classifies every one of the sibling's thirty-seven modules, and the layering test forbids reaching
sideways into the pipeline it was copied from.

What it **closed**, each having been a standing gap in this section:

- Every readout that was a per-epoch scalar off a CSV is now a per-recording quantity with a
  bootstrap interval over recordings: `pred_gap` in both estimators, the coupling readout, the three
  warm-up tertiles, the two source-lag warmth fractions and `kld_source_null`.
- The **availability-clock hazard is measured**, which nothing else could see: the source
  availability pattern is a deterministic function of $t$ shared by every row of a batch, so no
  permutation control can remove it. `source_null` reports
  $\Delta_{\mathrm{clock}} = \texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null}$ per
  recording, and `coupling_minus_clock_nats` reaches every arm table from the first run.
- The anchor geometry is a **FAIL-able verdict** rather than an assumption: `anchors_per_sample`
  must read $136$ and `target_warm_frac` exactly $1.0$ at the dense set, computed from the
  checkpoint's own geometry.
- The forecast is scored against three trivial predictors rebuilt in feature space, resolved by
  horizon step, and the observation model's calibration is checked per coefficient — so a block
  score is a log density of something.
- The lag readout is reported unbiased and the truncation is **measured** rather than assumed:
  preflight records `lag_support_margin_steps` $= F - (L - 1) - \texttt{lag\_floor} = 43$ at the
  shipped geometry, and `lag_kl` and `attention` read that number rather than hard-coding the
  simplification an arm could invalidate.
- The channel axis is used as the frequency axis it is: `spectral_skill` resolves the forecast gap
  by the band of the target coefficient, joined through a persisted kept-axis channel map.
- **The lag question is asked interventionally as well as observationally.** `occlusion` zeroes the
  stored source coefficients in configured lag bands relative to one scored anchor per segment,
  leaves the availability announcement untouched, re-encodes through the run's own K/V path and
  reports the per-horizon-step NLL change on a common anchor support with common random numbers. It
  answers *when did the source matter* on an axis the lag window's near edge cannot pin, and it is
  the third intervention arm beside the permutation and null ones —
  `controls.occluded_forward_outputs`, reusing their shared attend-and-pose tail.
- **The lag question is asked of the anchors that carry the coupling, and in forecast space.**
  `lag_high_kl` selects anchors by a quantile band of the pooled per-anchor KL (upper 30%, its
  complement, upper 10%) off a third collection sidecar, `per_anchor_vectors.npz`, which holds the
  pooled KL attribution and the head-averaged attention at every contributing anchor; it resolves
  the selection's lag profile and share against both clinical clocks, and scores every band by the
  Monte Carlo forecast gain of the same anchors -- the high band against the rest band paired within
  recording, a gain-selected band and its overlap with the high band, the gain by KL decile and by
  argmax lag, and the attribution share per geometry band against the occlusion cost. That is the
  reading that separates "the source moved the belief" from "the forecast got better where it did".
- **The censoring edges are read as censoring.** `check_argmax_lag`'s near edge is
  `min(attainable)` off the per-lag anchor counts rather than the literal $0$, symmetric with the
  far edge; a pin there reports INCONCLUSIVE with the physical-lag arithmetic in its message, and
  whether the machinery is alive is judged from the shape vocabulary (`lag_shape`'s degeneracy flag,
  the peak width, the mass above half the peak) and the per-head entropies rather than from the
  argmax. A degenerate profile still FAILs at either edge or in the middle.
- **The per-head profiles are on the page**, beside the pooled one, with each head's argmax, peak
  width, mass above half peak, near and far mass, attention entropy and KL share. That is the
  surface an alignment-arm or K/V-arm comparison is expressible on; the pooled profile is not.
- **The run's arm is on the page**: the configured `causal_align_reference` label, the configured
  source reference, both resolved clocks, the inter-stream offset and `lag_kv_source`, printed on
  the same console block as the delay and carried in `summary.json` with the *configured*, *built*
  and *resolved* readings separated — so a config naming one arm while the checkpoint carries
  another is visible rather than merged away.
- **`clock_margin_min_nats` is set**, at $0.15$ nats, so `coupling_exceeds_availability_clock` is a
  real gate and the acceptance set is ten criteria rather than nine. The provenance is stated beside
  the value in both cfs cells' `eval_overrides.yaml`: the diagnosed unaligned run's observed
  $\Delta_{\mathrm{clock}} = 0.160$ with interval $[0.157, 0.164]$. The margin's provenance is the
  unaligned arm; the gated quantity is right on both arms.

What it deliberately does **not** do, so each is a known absence rather than an oversight:

- **No phase-domain analysis.** `coherence` and `spectra.py` are not ported at all. A stored
  coefficient is a *modulus*: the analysing filter's phase was discarded before the value was
  written, so phase agreement, group delay and the residual's three-way split into irreducible,
  timing and amplitude terms have no analogue here at any window length. `spectral_skill` replaces
  the half that does exist and is named differently on purpose. Reviving the raw construction is not
  the fix: a $\tau$-slice on this grid gives $136$ samples at $0.25$ Hz per channel, Nyquist
  $0.125$ Hz, over band-limited envelopes rather than one trace.
- **No deceleration forecast skill and no contraction-triggered response.** Both score a clinical
  heart-rate trace in beats per minute; this model forecasts $98$ coefficients, and defining a
  deceleration on a channel axis with no order and no clinical unit is a new construction rather
  than a port. Contraction-*conditioned* coupling is unaffected and ships, because it conditions on
  the timing rather than on the forecast's shape.
- **No clinical unit anywhere.** Every readout stays in the loader's $z$ units, labelled
  `normalised`. Inverting the per-channel statistics would put the $98$ scored channels on scales
  spanning orders of magnitude, which destroys every pooled statistic and the tertile split with it.
- **No physical-lag readout**, for the reason the second `lean-limit:` above already states,
  narrowed rather than withdrawn — and the evaluation adds only that the caveat now travels on every
  lag-resolved artifact rather than living in this document alone.
- **No cross-target arm table against `lag_attn_fs`.** The blocks still differ ($2940$ against
  $2340$ coefficients) even though the horizons no longer do, so a level comparison would invite
  exactly the reading §5 forbids. The cross-cell table carries `lag_attn_transformer_cfs` only,
  where both cells sum the same block over the same anchor count under the same objective.
- **No keep-mask band-sufficiency analysis.** Porting the raw pipeline's `lag_ablation` would need a
  model-side `lag_band_mask` forward seam this family does not have, and under encoder-state K/V it
  reads short-band-sufficient by construction. The occlusion analysis is the instrument chosen
  instead, and it additionally resolves by horizon step.
- **The occlusion readout is a headline scalar and deliberately not a verdict.** How much a forecast
  loses without a band of source is a measurement whose healthy range nothing has established, and a
  threshold guessed before the first production runs would decide a pass or a fail on exactly the
  run that was going to measure it. Four entries reach the arm table rather than one — the band
  name, its delta, its peak horizon step and its **live fraction** — because a band deep in the
  warm-up scores near zero for a reason that is about the geometry rather than about the source.
- **The evaluation applies neither objective weight.** §5. Its `nll_*` and `pred_gap` are true log
  densities, so they are not the training path's numbers under either weight, and
  `horizon_weight_halflife_steps` is deliberately absent from the binding's `GEOMETRY_KEYS` for
  exactly that reason: no evaluated readout applies it. `prior_availability_input`, `lag_kv_source`
  and `persistence_residual` **are** in that tuple, because each changes what an evaluated number
  means — the first what the KL is, the second what the lag map is over, the third what the
  predictor is.

## 15. Deviation record

Where the built package differs from the design it was built from, and why.

- **The target domain is two mixins, not one.** The input half (`CausalWarmupInputs`) and the target
  half (`CausalFeatureForecastTarget`) are separate objects because only the second is a
  `FeatureForecastTarget` subclass; folding the forward and the adapter override into it would put
  encoder-facing code in a class whose whole property is that it mentions no encoder.
- **The eleven resolved readouts, not six.** `_resolved_forecast_gaps` returns the parent's four
  splits, the three warm-up tertiles and the four this package adds on the net side. The parent's
  four are **recomputed** rather than inherited, and only their mask changes: the reduction is the
  parent's own `_forecast_gaps_from_mask`, so each stays a partial sum of the gap beside it.
- **`anchor_stride` defaults to $1$, not to $H$.** The inert value, so a model constructed without an
  opinion behaves like the rest of the family; the tiling is a configuration decision and every
  shipped config states it. `tests/test_config_load.py` asserts `anchor_stride == horizon` in the
  default and in every arm rather than leaving the pairing to be noticed.
- **The gradient clip moved and the additive margin moved; the other two loss-scale constants did
  not, and that is a measurement.** `gradient_clip_val` ships at $6000$ against the two-sided
  sibling's $5000$ (the smallest round value above a measured $q_{99} = 5742$), and
  `additive_margin` at $3.0\mathrm{e}{+3}$ against $5.0\mathrm{e}{+3}$ — $\approx 1.9\times$ the worst
  excursion above the breaker's own EMA in the noisiest regime the fixture can produce. `ema_floor`
  stays at $1.0\mathrm{e}{+9}$ because it is a *switch* rather than a scale, and `horizon_embed_std`
  stays at $0.8$ because the quantity it is chosen against — the post-initialisation correlation
  between two horizon tokens — is a per-pair quantity that does not depend on how many tokens there
  are: measured at $0.445915$ for $H = 15$ against $0.447476$ for $H = 30$. `RESULTS.md` carries the
  percentiles.
- **`horizon_depth` stays at $4$ although $3$ would fit the horizon.** Depth $3$'s receptive field is
  $15$ tokens, which fails the family's recorded $\mathrm{RF} \ge H + 1$ criterion by exactly one at
  this horizon — and the argument is moot in both directions anyway, because
  `horizon_attention_blocks: 2` mixes every horizon token unmasked after the refine stack. What
  settles it is parity: keeping the sibling's depth removes a confound from both comparisons, and the
  only thing depth $3$ buys is $328{,}960$ parameters. `sweep_horizon_depth_3.yaml` ships the arm.
- **$\beta = 1.0$ / $\beta_p = 0.1$ is carried across unmeasured at this scale**, and the config says
  so rather than implying otherwise. That pair was selected by a four-arm sweep over a block of
  $2340$ coefficients at $\approx 240$ decoded anchors per step; this block is $2940$ at
  $\approx 4.6$. The sweep came out **monotone** with its lower edge winning every column, which is
  why the lower edge is what transfers rather than a scale-matched rescaling — the one prediction
  that sweep refuted was exactly the scale-matching argument. Resolved by the headline run's
  `kld_active_frac` and `logvar_prior_floor_frac`.
- **`causal_reach_budget_s` is present and required `null`.** Not merely unnecessary but undefined
  here: it prunes channels by the forward reach of a *two-sided* Morlet, measured on a bank that did
  not produce these coefficients, and a delay is a shift. The resolver refuses it alongside a warm-up
  budget by name. It is present rather than absent so the parity comparison against the two-sided
  sibling stays leaf-for-leaf.

### What moved in this revision, against the record it replaces

Seven changes, each one against a statement the pre-revision record made. They are listed as
deviations rather than folded into the sections above, so a reader holding the earlier record can
see exactly which of its sentences no longer hold.

- **The lag attention's keys and values are no longer the deep source state**, and the deep source
  encoder is no longer built at all. The earlier record described $h_u$ as what the attention reads
  and as a module every arm carries; §8.1 replaces both statements, and §13's off-state row is where
  the earlier parameter totals still live.
- **The prior is conditioned on a function of $t$.** "The prior never sees the source" is restated
  as "the prior sees no function of the source *values*" (§8.2). The source-purity tests assert the
  restated form; nothing about the forecast claim or the coupling readout rested on the stronger
  one, and the mechanism's off-state is bitwise the prior the earlier record described.
- **The decoder takes a second tensor.** The earlier record's "the decoder receives the latent and
  nothing else, so there is no bypass to report" becomes "the latent and a **target-only**
  persistence vector"; the no-bypass property is unchanged and is now a property of what that tensor
  is rather than of there being only one (§5).
- **The reconstruction is weighted on the horizon axis as well as the channel axis** (§5). The
  earlier record's caution that a weighted block is not a log-density applies to both weights, and
  the evaluation still applies neither.
- **The source stream is aligned onto its own clock** (§3.1). The earlier record described one
  reference applied to both streams; the source now has a second, the inter-stream offset is a known
  constant that travels through the lag arithmetic, and the source keep-index narrows $47 \to 39$.
- **The lag bias ships seeded flat.** `alibi_slope_scale: 0.0` against the constructor default
  $1.0$. No code moved; the shipped configuration did, and the decaying seed is a named arm.
- **The training controls are on.** Early stopping is enabled at patience $50$ on `val/total_loss`,
  and a second `ModelCheckpoint` on `val/nll_full_block` is built behind a single optional monitor
  key, because the composite optimum and the best conditioned forecast are different epochs.
  `RESULTS.md` registers both epochs as things a run reports.

## 16. Running it

From the repository root.

```bash
# Production, 7 ranks. TEB_RUN_STAMP is required so ranks 1..N-1 share rank 0's run directory,
# and the rank count must equal len(general_config.cuda_devices).
TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
    -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/default.yaml

# Local smoke: one epoch, one device, the committed causal fixture.
python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/tiny.yaml

# Dev-box validation: the shipped geometry over a causal HIE sample shard.
python -m teb_vae.lag_attn_cfs.trainer --config teb_vae/lag_attn_cfs/configs/smoke_hie.yaml

# Can this architecture recover a delay it is known to be looking at? Fits the committed
# planted-delay fixture for a few epochs and reads the lag profile back through the evaluation's
# own code. Minutes on a dev box; --mode manifest builds the model and stops.
python teb_vae/lag_attn_cfs/lag_recovery_check.py \
    --config teb_vae/lag_attn_cfs/configs/planted.yaml

# Price a candidate source alignment reference before pinning one: surviving channels by block,
# the freshest-source recency at the anchor, and where a physiological delay lands in the window.
python teb_vae/lag_attn_cfs/warmup_budget.py

# Score a finished run against the criteria RESULTS.md registered, while it is still in flight.
python -m teb_vae.lag_attn_cfs.check_run --run-dir <run>

# Evaluate a finished checkpoint on the held-out population: one reviewable directory.
python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <run>/model_checkpoints/<name>.ckpt

# Gate that directory offline, on a box with no torch installed.
python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json
```

`RUN_CONFIG` near the bottom of `trainer.py` names the config used when the module is launched with
no command line, so the entry point works from an IDE's Run button with the only operator action being
to edit a value inside the file; a `--config` on the command line always wins, and a relative path
resolves against the repository root rather than the working directory. `check_run.py`, `eval/run.py`,
`eval/probe.py`, `eval/verify.py`, `lag_recovery_check.py` and `warmup_budget.py` follow the same
convention through their own `RUN_ARGS` dicts. Note that a Run-button launch of `default.yaml` is a
*single* process whose seven `cuda_devices` make the framework spawn DDP workers underneath it.

**The named comparison arms are launched exactly like the default, with their own config**, and
`RESULTS.md` lists them by file. Each is the default plus one named leaf plus the two identity keys
(`run_name`, `tags.variant`) that put the arm in the run's own directory name — which is the guard
against a run being misattributed to the arm someone meant to launch rather than the one they did.

**`check_run.py` and `eval/verify.py` answer two different questions and neither substitutes for the
other.** The first reads `train_results/metrics_history.csv` and needs no checkpoint, no shard and no
`torch`: it answers *did the fit behave*, in-sample, per epoch, with no denominator and no interval,
and it answers it while a run is still going. The second reads `eval_results/summary.json` and needs
the same nothing: it answers *is this finished checkpoint acceptable*, on a held-out population, per
recording, with intervals. §14 and `eval/EVAL.md` both carry the cross-reference.

The gate:

```bash
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m "not slow"
.venv/Scripts/python.exe -m pytest teb_vae/lag_attn_cfs/tests -q -m slow
```

---

## 17. The switch inventory

Six mechanisms, each a key whose **off-state reproduces the pre-revision model bitwise**. That is
the safety rail rather than a convenience: it is what makes any single mechanism removable by one
YAML line without reverting code, what makes the arms comparable, and what the construction tests
assert against — state-dict key for state-dict key, parameter for parameter.

| Key | Shipped | Off-state | Level | What it decides |
|---|---|---|---|---|
| `lag_kv_source` | `conv_stem` | `encoder` | constructor | which source representation the lag attention's keys **and** values are built from, and whether the deep source encoder is built at all (§8.1) |
| `prior_availability_input` | `true` | `false` | constructor | whether the prior head is conditioned on the source pathway's encode of silence (§8.2) |
| `persistence_residual` | `true` | `false` | constructor | whether the decoder's mean head carries a target-only persistence term (§5) |
| `horizon_weight_halflife_steps` | `15.0` | `null` | constructor | whether the reconstruction weights the horizon axis, and with what half-life (§5) |
| `alibi_slope_scale` | `0.0` | `1.0` | constructor | whether the learnable lag bias is seeded flat or with a monotone decay towards lag $0$ (§8.1) |
| `causal_align_reference_source` | `288.2672` | `null` | task | whether the source stream is aligned onto its own faster clock or shares the target's (§3.1) |

**Three properties of the table are load-bearing.**

**An unknown value is refused by name.** `lag_kv_source` names its three admissible values in the
refusal; a source reference that snaps to no stored source delay names the delay it was compared
against; a source reference set against an unaligned target is refused because the scheme is a pair
of clocks and one of them would be missing. A silently accepted value is the failure mode every one
of these refusals exists to prevent.

**A key absent from a cell's constructor is dropped in silence.** The driver builds a run's kwargs
by sweeping `inspect.signature(MODEL_CLS.__init__)`, so a key present in the parent and forgotten in
this cell's own signature would leave the arm training as the baseline with nothing in its log
saying so. Every one of the five constructor keys is therefore written out in this cell's signature
as well as the parent's, and the evaluation console prints the *configured* value beside the
*built* one so the two readings can disagree visibly (§12).

**The two levels are not interchangeable.** The five constructor keys land in the checkpoint's
`model_kwargs` and are what a checkpoint is rebuilt through; `causal_align_reference_source` is
*resolution* — it decides which channels exist and by how much each is shifted, before any
constructor is called — so it reaches a run through the resolved warm-up budget and its consequences
are stamped as the shift vectors and the keep-index rather than as a key.

**Where each off-state is pinned.** The construction pins live in `tests/test_construct.py` and the
shared owners' stability file; the horizon weight's off-state is pinned as **loss equality** rather
than as construction, because a construction pin over a switch that builds no parameter is vacuous;
the prior head's `clock=None` equality is pinned in the heads owner file; and the decoder's
one-tensor invariant is restated to admit exactly the target-only persistence argument and nothing
else.

---

## Amendment (2026-09): the forecast-target clock

Everything above describes the target as *gathered, never delayed*: the encoder inputs aligned
onto a common content clock, the scored element left at each channel's own stored index. That
asymmetry had a cost the design carried knowingly — the scored horizon elements describe physical
content at $(t{+}1{+}\tau)\Delta - \kappa\tau_c$, per-channel instants spanning ~13 s to ~350 s in
the past, so the slowest kept channel's "forecast" draws $\nu = 0.026$ of its value from the
horizon window and is near-copyable. The latent was never forced to encode genuinely predictive
information at the slow scales. `causal_target_forecast_clock` makes the clock of the *question* a
per-run choice, and the shipped default moves it.

**One signed shift, three clocks.** The resolver turns the key into `target_forecast_shift`, one
signed integer $s_c$ per kept target channel; the scored element at anchor $t$, horizon step
$\tau$, kept channel $c$ reads stored step $t + 1 + \tau + s_c$.

- **`physical`** (shipped): $s_c = \operatorname{round}(\kappa(\tau_c - \tau_{\min})/\Delta) \ge 0$
  with $\tau_{\min}$ the fastest kept channel (13.3405 s, shift exactly $0$). Every horizon step
  scores content at one physical *future* instant; every element is a strict forecast for every
  channel; novelty becomes near-uniform. Advancing is admissible on a target where it is refused
  on an input: the target is what the anchor is asked to predict, so a later stored step is a
  strictly harder question, not a leak.
- **`input`**: $s_c = -d_c$, the continuation of the encoder's own aligned stream. Kept as an arm:
  the scored content sits up to ~350 s in the past and the source clock is ~100 s fresher than it,
  so `pred_gap` measures contemporaneous inference there rather than directed prediction.
- **`stored`** (and the absent key): $s_c \equiv 0$, byte-for-byte the historical cell — kwargs,
  tensors, checkpoints, and the resolver's output all unchanged.

**What generalises, mechanically.** The floor's scored-target half becomes
$F \ge \max_c(W'_c - s_c) - 1$ (the input-warmth half is untouched, so `warmup_period: 134`
stands); the anchor ceiling becomes $T_{\mathrm{valid}} - \max(0, \max_c s_c)$ — under `physical`
the largest advance is 85 steps, dense anchors 136 → 51, and `anchor_stride` moves to 5 to keep
~10 training tiles per sample ($A_{\max} = 11$, the supervision density the $H = 15$ geometry
trained at; it shipped briefly at 10, ~5–6 tiles, before being halved to restore that density);
the persistence gather clamps to
$\min(s_c, 0)$, which removes an exact per-channel copy under `input` (the anchor's own value *is*
the scored element at $\tau = d_c - 1$) and is what keeps the residual causal under `physical`;
and every mask is built from `scored_weight`, the validity signal min-pooled over the shift span —
conservative, 3-D, anchor-count denominators intact (lean-limit recorded on the method). The
`_build_forecast_target` override delegates to the parent whenever no shift is set, so the
two-sided cells and every stored-clock checkpoint are bitwise untouched; the shift tuple is
emitted into `model_kwargs` only when the clock is not `stored` (`FORECAST_ALIGN_MODEL_KWARGS`),
and the evaluation preflight compares it between config and checkpoint like the other six tuples.

**What the lag axis means under `physical` — a re-registration.** The 20–60 s proximate UP→FHR
band is *structurally* censored: the offset the lag identity subtracts becomes
$\tau^u_{\mathrm{ref}} - \tau_{\min}$, and clearing lag 0 at the far horizon step would need
$\tau^u_{\mathrm{ref}} < 34$ s — below every stored source delay. That is arithmetic, not tooling:
a strict forecast conditions only on the past, and the proximate driver of a predicted instant is
unrecorded at the anchor, for every reference. What past UP carries about future FHR is the
contraction cycle's phase, so the axis reads the **recurrence** of the 2–5 min IUP rhythm: at the
retained source clock (288.2672 s — kept, because any clock fast enough to help loses the
envelope block) and `max_lag: 90`, the every-horizon-step readable physical-delay window is
$[374.9, 618.9]$ s, the previous one-to-two contractions. `warmup_budget.py` prints this window
per candidate when run against a physical-clock config. The pre-registered reading "alignment
should not improve `pred_gap`" holds only within stored-clock arms; re-clocking the question is
expected to move it.

**What is deliberately unchanged.** The novelty- and warm-tertile partitions keep their
stored-clock rankings (readouts, not losses; under `physical` the novelty split reads as a
partition by stored-clock novelty rather than as a measurement on the scored clock).
`planted.yaml` pins `stored` — its planted delay is stamped in stored steps, and
`lag_recovery_check` refuses any other clock by name. The arms are
`sweep_target_clock_stored.yaml` (today's target *and* tiling, exactly) and
`sweep_target_clock_input.yaml`.

## Amendment (2026-09): the diagnostic page's clocks

§11.1 describes the input rows as "the streams as the encoders receive them", and the page's
footnote as first written stated that those rows sit $\kappa\tau_{\mathrm{ref}}$ to the *right* of
the raw traces. That was a labelled misalignment rather than a corrected one: an aligned channel at
step $t$ carries content centred at $t\Delta - \kappa\tau_{\mathrm{ref}}$ — $351.9$ s earlier on
the shipped target clock, $252.2$ s on the source's — so a deceleration in the raw FHR appeared in
the `fhr_st` row six minutes later, and the same channel sat at two different places on the
`input_target` row and the `pred_truth` row. The page now draws every re-clocked stream on the
physical axis the raw row sets:

- **The input rows** are shifted left by $\kappa\tau^y_{\mathrm{ref}}$ and
  $\kappa\tau^u_{\mathrm{ref}}$ (`InputStreamPanel.content_offset_s`, zero on the two-sided pages,
  whose coefficients are stamped with the instant they describe), the warm-up staircase with them,
  and the last $\kappa\tau$ seconds are hatched: content the encoder has not received by the
  segment's end.
- **The forecast rows** — the lanes, the five field rows and the per-window score — are shifted by
  $\kappa\tau$ of the *scored* clock, `WarmupBudget.target_forecast_clock_delay_s`: $\tau_{\min} =
  13.34$ s under the shipped `physical` clock ($11.7$ s, under three steps), $\tau^y_{\mathrm{ref}}$
  under `input`, and none under `stored`, where no constant exists and the rows stay at step index
  with the axis saying so.
- **The anchor marks and the latent, KL and lag rows stay at the anchor step**, which is the
  instant a forecast is made from. The gap between an input row's hatched tail and the anchor
  floor is the staleness §3 prices, now visible rather than footnoted.

Both $\tau$ come from the resolved budget, which the task binds into the page seams; the net stamps
only the per-channel step shifts, from which no $\tau$ is recoverable. The evaluation runner
therefore attaches the re-resolved budget to the loaded task after preflight
(`eval/run.py::attach_warmup_budget`). Before this amendment it never did, so every evaluation page
was drawn at step index with no clock stated on it, while the training callback's pages carried
the footnote.
