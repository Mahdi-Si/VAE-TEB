Here’s a clean way to marry **TEB** (to capture *directed* info $X\!\to\!Y$) with **β-TCVAE** (to make the latent factors independent):

### Recommend splitting the latent

Use two parts of the latent at time $t$:

* $z^{Y}$: “intrinsic” factors of $Y$ (what $Y_{1:t}$ already explains)
* $z^{X\to Y}$: “extrinsic” factors transferred from $X$ that help predict $Y_{t+1:t+H}$

Encoders and prior

$$
\begin{aligned}
q_\phi(z^{Y}\mid Y_{1:t}),\qquad
q_\phi(z^{X\to Y}\mid X_{1:t},Y_{1:t}),\qquad
r_\psi(z^{X\to Y}\mid Y_{1:t}) \ \text{(TEB conditional prior)}.
\end{aligned}
$$

Decoder $p_\theta(Y_{t+1:t+H}\mid z^{Y}, z^{X\to Y}, Y_{1:t})$.

---

## Combined loss

$$
\begin{aligned}
\mathcal{L} \;=\;
&\underbrace{\mathbb{E}_{q_\phi}\big[-\log p_\theta(Y_{t+1:t+H}\mid z^{Y}, z^{X\to Y}, Y_{1:t})\big]}_{\text{prediction (max }I(z;\,Y_{\text{future}}\mid Y_{\text{past}})\text{)}} \\[2mm]
&+ \lambda_{\mathrm{TE}}\;\underbrace{\mathrm{KL}\!\big(q_\phi(z^{X\to Y}\mid X_{1:t},Y_{1:t})\ \|\ r_\psi(z^{X\to Y}\mid Y_{1:t})\big)}_{\text{TEB bottleneck } \approx I(z^{X\to Y};X_{\text{past}}\mid Y_{\text{past}})\ \text{upper bound}} \quad \text{\small\cite{}} \\[1mm]
&+ \beta_Y\;\underbrace{\mathrm{TC}\!\big(q(z^{Y})\big)}_{\text{disentangle intrinsic factors}} 
\;+\; \beta_X\;\underbrace{\mathbb{E}_{Y_{1:t}}\big[\mathrm{TC}(q(z^{X\to Y}\!\mid Y_{1:t}))\big]}_{\text{disentangle transferred factors (conditional TC)}} \quad \text{\small\cite{}}\\[1mm]
&+ \gamma_Y\;\sum_j \mathrm{KL}\!\big(q(z^{Y}_j)\,\|\,p(z_j)\big)\;+\; \gamma_X\;\sum_j \mathrm{KL}\!\big(q(z^{X\to Y}_j)\,\|\,p(z_j)\big) \\[1mm]
&+ \eta\;\underbrace{I_q\!\big(z^{Y}; z^{X\to Y}\mid Y_{1:t}\big)}_{\text{optional cross-talk penalty (often set to 0 if TC over }[z^{Y},z^{X\to Y}]\text{ used)}}.
\end{aligned}
$$

* **TEB term** (second line): this is the variational *bottleneck* that compresses the *directed* info from $X$ into the extrinsic latent, implemented via a **conditional prior** $r_\psi(z^{X\to Y}\!\mid Y_{1:t})$. It’s the practical upper bound used in TEB for $\,I(z^{X\to Y};X_{\text{past}}\mid Y_{\text{past}})$. ([ar5iv][1])
* **TC terms** (third line): these are the β-TCVAE pieces. For $z^{Y}$ we use standard **total correlation** on the aggregated posterior $q(z^{Y})$. For $z^{X\to Y}$ we recommend a **conditional TC**—penalize dependence among the extrinsic coordinates *given the context $Y_{1:t}$*, using $\mathbb{E}_{Y}[\mathrm{TC}(q(z^{X\to Y}\!\mid Y))]$. This mirrors the β-TCVAE rationale (TC drives disentanglement) while respecting TEB’s conditional structure. Use minibatch-weighted sampling (MWS) to estimate TC efficiently. ([NeurIPS Papers][2], [Wikipedia][3])
* **Dim-wise KLs** (fourth line): as in β-TCVAE with $\alpha=\gamma=1$, these keep marginals near the prior and stabilize scale; leave $\gamma_X,\gamma_Y$ at 1 unless you have a strong reason to change them. ([NeurIPS Papers][2])
* **Cross-talk** (last line, optional): if you see leakage between $z^{Y}$ and $z^{X\to Y}$, add a small $\eta>0$ or, alternatively, **replace the two separate TC terms by a single $\beta\,\mathrm{TC}(q([z^{Y},z^{X\to Y}]))$** which automatically discourages dependencies across the blocks (often sufficient).

---

## Why this makes sense

1. **Keep what helps $Y$ and only what comes from $X$**
   The TEB KL-to-conditional-prior is the principled way to *minimize* $I(z^{X\to Y};X_{\text{past}}\mid Y_{\text{past}})$ while the reconstruction *maximizes* $I(z;\,Y_{\text{future}}\mid Y_{\text{past}})$ (the CMNI point). That’s exactly what you want for directed information transfer. ([ar5iv][1])

2. **Make the codes interpretable**
   β-TCVAE showed that **penalizing TC** (not “the KL as a whole”) is what drives disentanglement; setting $\alpha=\gamma=1$ and tuning only $\beta$ is effective. We import that idea: penalize TC *within* each subspace (or jointly) so each factor lives in its own coordinate. Use **MWS** to estimate $q(z)$ from a minibatch with no extra networks. ([NeurIPS Papers][2])

3. **Respect conditionality**
   Because $z^{X\to Y}$ is defined *given* $Y_{1:t}$, using **conditional TC** for that block (i.e., $\mathbb{E}_Y[\mathrm{TC}(q(z^{X\to Y}\!\mid Y))]$) is the right analogue of β-TCVAE in TEB’s setting (independence of extrinsic factors *after conditioning on $Y$*). The notion of conditional TC is standard. ([Wikipedia][3])

---

## Practical notes

* **Weights**: start with $\lambda_{\mathrm{TE}}\in[1,4]$, $\beta_Y\in[2,8]$, $\beta_X\in[1,4]$, $\gamma_X=\gamma_Y=1$. If you collapse $z^{Y}$ and $z^{X\to Y}$ into a single $z$, just use one $\beta$ on $\mathrm{TC}(q(z))$.
* **Estimating TC**: follow β-TCVAE’s **minibatch-weighted sampling** for $\log q(z)$ and $\log\!\prod_j q(z_j)$; it’s simple, fast, and avoids an auxiliary discriminator. ([NeurIPS Papers][2])
* **Why not only β-TCVAE on $[X,Y]$?** That would promote independence but wouldn’t isolate *directional* $X\!\to\!Y$ information; TEB’s conditional prior is what enforces the directed bottleneck. ([ar5iv][1])

If you’d like, I can sketch PyTorch code for the **conditional-TC** estimates (it’s just β-TCVAE’s MWS computed per context minibatch) and the **TEB KL** to $r_\psi(z^{X\to Y}\!\mid Y)$.

[1]: https://ar5iv.org/pdf/2211.16607 "[2211.16607] Transfer Entropy Bottleneck: Learning Sequence to Sequence Information Transfer"
[2]: https://papers.neurips.cc/paper/7527-isolating-sources-of-disentanglement-in-variational-autoencoders.pdf "Isolating Sources of Disentanglement in Variational Autoencoders"
[3]: https://en.wikipedia.org/wiki/Total_correlation?utm_source=chatgpt.com "Total correlation"
