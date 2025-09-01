Yep — you’re talking about the **Transfer Entropy Bottleneck (TEB)**, which you can implement in a VAE-style way.

**What it is (in one line):**
TEB is an information-bottleneck method that forces a model to keep only the **directed information** flowing from a *source* time series $X$ into a *target* time series $Y$ when predicting the future of $Y$. Formally, the directed info you care about is **transfer entropy**:

$$
\mathrm{TE}_{X\to Y} = I\!\big(X_{\text{past}};\, Y_{\text{future}} \,\big|\, Y_{\text{past}}\big),
$$

i.e., how much knowing the past of $X$ reduces uncertainty about the future of $Y$, beyond what $Y$’s own past already tells you.

**How the TEB objective looks in a VAE form:**
TEB uses an encoder–decoder with a **conditional prior** so that the latent $z$ carries only the part of $X$’s past that truly helps with $Y$’s future:

* Encoder: $q_\phi(z \mid X_{1:t}, Y_{1:t})$
* Conditional prior (“context prior”): $r_\psi(z \mid Y_{1:t}) \approx p(z \mid Y_{1:t})$
* Decoder: $p_\theta(Y_{t+1:t+H} \mid z, Y_{1:t})$

A practical loss (TEB-VAE) is:

$$
\mathcal{L} \;=\; 
\underbrace{\mathbb{E}_{q_\phi}\!\big[-\log p_\theta(Y_{t+1:t+H}\mid z, Y_{1:t})\big]}_{\text{maximize } I(z;\,Y_{\text{future}}\mid Y_{\text{past}})}
\;+\;
\beta\,\underbrace{\mathrm{KL}\!\big(q_\phi(z\mid X_{1:t},Y_{1:t}) \,\|\, r_\psi(z\mid Y_{1:t})\big)}_{\text{minimize } I(z;\,X_{\text{past}}\mid Y_{\text{past}})}
$$

This mirrors the original TEB derivation: maximize the conditional information your latent has about $Y$’s future, while **bottlenecking** the conditional information it keeps from $X$’s past. The KL to a *learned conditional prior* is the key trick; it upper-bounds the unwanted conditional mutual information so you only retain the truly transferred bits.

**Why it’s useful:**
In many real datasets the joint correlations inside $Y$ dominate; a plain joint model $[X,Y]\!\to\!Y_{\text{future}}$ can ignore small but crucial influences from $X$. TEB makes the $X\!\to\!Y$ transfer explicit and measurable, and even supports reusing a pre-trained $Y$-only model as the context network.

**Where it comes from:**
TEB extends the classic **Information Bottleneck** idea (compress $X$ while preserving info about $Y$) to the *conditional, directed* time-series setting, using variational bounds much like VIB/VAEs.
**Code / paper:**

* Paper: *Transfer Entropy Bottleneck: Learning Sequence to Sequence Information Transfer*.
* PyTorch reference implementation: `ximmao/TransferEntropyBottleneck`.

If you want, I can sketch this directly on top of your SeqVAE (define $q_\phi, r_\psi, p_\theta$ heads, wire the conditional KL, and add a simple dual update to target a desired TE budget).
