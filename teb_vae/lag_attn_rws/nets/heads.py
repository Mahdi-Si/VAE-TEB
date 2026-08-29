r"""The full-latent prior head: the complete target-only latent state, and nothing else.

The prior $p(z_t \mid Y_{\le t})$ here is not a regulariser on a side channel -- it *is* the
model's forecast state. The shared decoder receives only $z$, so everything the baseline
forecast knows about the target must pass through this head's two outputs. That is also why
there is no ``decoder_state`` output: a target-only conditioning path around the latent would
turn $z$ back into a residual code, and an unused head would leave dead parameters that a
distributed run must then be told to tolerate. The head is written without it rather than
reusing the sibling's ``PriorHead`` and discarding a tensor.

The pre-bound raw log-variance is returned alongside the bounded one because the posterior is a
residual on the *raw* value: the bound is a sigmoid and not idempotent, so a residual built on
the already-bounded value could not reproduce the prior exactly at zero delta, and the exact
zero-KL initialization would silently not hold.

**The head optionally conditions on a clock, and what that is not.** ``clock_dim`` builds a second
input path for a tensor that is a deterministic function of $t$ and the resolved configuration. It
carries **zero information about the source's values**: the composing model supplies it, and what
it supplies is the source pathway's response to a stream of exact zeros -- identical for every
recording in the dataset and identical under any intervention on the source.

The reason it is offered to a *target-only* head at all is that the posterior already sees it. The
source stream arrives over the first steps of a segment, channel by channel, and the source adapter
both masks and announces that; the encoder then carries the transient forward past the anchor floor.
Without the clock the KL between the two branches contains a term neither branch could have learned
from the data -- an availability clock, attributed to the source by every readout downstream.
Conditioning both branches on it lets that term cancel instead of being measured and subtracted.

**What this head does not decide is what the clock is.** It normalises and projects whatever it is
handed; the composing model chooses the tensor, and that choice is load-bearing. The instantaneous
availability staircase is the intuitive answer and is provably inert -- the anchor floor is refused
below the last step at which it changes, so every scored step sees the same constant vector.

The invariant this restates, in the only form that stays true: the prior sees no function of the
source's **values**. It was previously stated as "the prior never sees the source", which the clock
path does not violate and the sentence does not survive.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound


class FullLatentPriorHead(nn.Module):
    r"""Target-only prior $p_\theta(z_t \mid Y_{\le t})$ over the full latent.

    Produces three outputs from the target state $H^y$:

    * ``mu_prior`` -- prior mean ``(B, T, d_z)``, bounded by
      $\mu\_scale \cdot \tanh(\mathrm{raw} / \mu\_scale)$ so $|\mu^p| \le \mu\_scale$.
    * ``logvar_prior`` -- prior log-variance ``(B, T, d_z)``, smoothly bounded.
    * ``raw_logvar_prior`` -- the *pre-bound* log-variance ``(B, T, d_z)``, the base the
      posterior residual is applied to.

    Each head is fed through its own ``LayerNorm`` so the raw encoder state cannot drift
    unbounded through either of them.
    """

    #: Declared so the optional clock path types as its own classes rather than as
    #: ``Tensor | Module``, which is what ``nn.Module.__getattr__`` otherwise gives them. Both are
    #: ``None`` on a head built without a clock, matching the convention the availability adapter
    #: and the channel gates already use for a term that was not constructed.
    clock_norm: Optional[nn.LayerNorm]
    clock_proj: Optional[nn.Linear]

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 48,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        dropout: float = 0.1,
        mu_scale: float = 5.0,
        clock_dim: Optional[int] = None,
    ) -> None:
        """Initialize the prior head.

        Args:
            d_model: Encoder state width.
            d_z: Latent dimensionality.
            logvar_clamp: ``(lo, hi)`` effective range of the log-variance bound.
            dropout: Dropout used inside every internal MLP.
            mu_scale: Saturation magnitude of the tanh-bounded prior mean. Set large enough to
                be non-restrictive around the $N(0, I)$ reference.
            clock_dim: Width of the clock this head conditions on, or ``None`` -- the default --
                for a head that conditions on the target state alone and is bitwise the one built
                before this keyword existed. See the module docstring for what the clock is and is
                not; this head does not choose it.

        Raises:
            ValueError: If ``mu_scale`` is not positive, or if ``clock_dim`` is not positive when
                it is given at all. A zero-width clock is a head carrying a projection that can
                never be reached, which is a silent DDP hazard rather than an inert setting.
        """
        super().__init__()
        if mu_scale <= 0.0:
            raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
        self.logvar_clamp = logvar_clamp
        self.mu_scale = float(mu_scale)

        # Per-head input norms decouple the two heads from shared drift in h_y.
        self.mu_input_norm = nn.LayerNorm(d_model)
        self.logvar_input_norm = nn.LayerNorm(d_model)

        # The clock path: this head's OWN norm and projection, deliberately not shared with the
        # source adapter's ``mask_proj`` even though the two read the same pattern. What must be
        # shared is the pattern, not the map -- a shared projection would couple the gradients of
        # the target-only branch and the source pathway, which is precisely the coupling the KL
        # between them is supposed to measure.
        #
        # ``bias=False``, like ``mask_proj``: with the weight zeroed the output is then EXACTLY
        # zero whatever the norm's affine does, which is what makes the zero-KL start exact rather
        # than approximately true. Built conditionally and applied unconditionally-when-built --
        # the family's DDP rule, so no batch can leave the projection out of the graph.
        self.clock_norm = None
        self.clock_proj = None
        if clock_dim is not None:
            if int(clock_dim) <= 0:
                raise ValueError(
                    f"clock_dim must be > 0 when given, got {clock_dim}. Pass None for a head "
                    f"that conditions on the target state alone; a zero-width clock builds a "
                    f"projection no forward can reach."
                )
            self.clock_norm = nn.LayerNorm(int(clock_dim))
            self.clock_proj = nn.Linear(int(clock_dim), d_model, bias=False)
            self.zero_init_clock()

        self.mu_prior_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.logvar_prior_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def zero_init_clock(self) -> None:
        """Zero the clock projection, so the head starts exactly where it starts without one.

        Called from the constructor and **again** from the composing model's post-initialisation
        block: the generic :func:`~utils.initialization` pass xavier-fills every ``nn.Linear``
        after the modules are built, so a constructor-only zero is silently refilled and the exact
        zero-KL start goes with it. Idempotent, and a no-op on a head with no clock path.

        Initialisation only. Calling it on a trained model would discard exactly what the clock
        path learned.
        """
        if self.clock_proj is not None:
            nn.init.zeros_(self.clock_proj.weight)

    def forward(
        self, h_y: torch.Tensor, clock: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce the complete target-only latent distribution.

        Args:
            h_y: Target history state ``(B, T, d_model)``.
            clock: The clock, ``(T, clock_dim)`` or ``(1, T, clock_dim)`` or
                ``(B, T, clock_dim)`` -- anything that broadcasts against ``h_y`` -- required
                exactly when the head was built with a ``clock_dim`` and refused otherwise. It is
                added to the head input **ahead of** the two input norms, so both heads see it
                through their own normalisation rather than one of them seeing a pre-normalised
                sum.

        Returns:
            ``(mu_prior, logvar_prior, raw_logvar_prior)``, each ``(B, T, d_z)``, with
            ``logvar_prior == smooth_bound(raw_logvar_prior)`` exactly.

        Raises:
            ValueError: If the clock and the head disagree about whether there is one. Both
                directions are refused rather than tolerated: a head built with a clock and called
                without one would leave its projection out of the graph on that step -- the
                ``find_unused_parameters=False`` hazard -- and a clock handed to a head that cannot
                use it would be silently discarded while every readout downstream reported a model
                that had cancelled the availability term.
        """
        if (self.clock_proj is None) != (clock is None):
            raise ValueError(
                f"the prior head was built {'with' if self.clock_proj is not None else 'without'} "
                f"a clock path and was called {'with' if clock is not None else 'without'} a "
                f"clock. The two are one decision: prior_availability_input builds the projection "
                f"and the forward that supplies it, and half of that is a model whose KL means "
                f"something other than what its configuration says."
            )
        # Structurally the original computation when there is no clock path -- the same two norms
        # over the same tensor -- so a model at the constructor default is bitwise the one built
        # before this keyword existed.
        features = h_y
        if self.clock_proj is not None and self.clock_norm is not None:
            features = h_y + self.clock_proj(self.clock_norm(clock))

        raw_mu = self.mu_prior_head(self.mu_input_norm(features))
        mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)

        raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(features))
        logvar_prior = smooth_bound(raw_logvar_prior, *self.logvar_clamp)
        return mu_prior, logvar_prior, raw_logvar_prior
