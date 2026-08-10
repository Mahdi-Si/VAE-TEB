r"""The training task: the sibling's, with one seam re-pointed at the feature stream.

Everything about turning a batch into a loss is inherited from
:class:`~teb_vae.lag_attn_rws.task.SeqVaeLagAttnRwsTask`, and that is the design rather than an
economy. Two models are only comparable if they optimise the same thing, so the objective

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p,
$$

its $\beta$ schedule, its metric surface, the validation-only permutation control, the
spike-breaker wiring, the pre-clip gradient-norm logging and the checkpoint contract are the same
code here, not a copy of it. That parent names no model class and no target domain -- it asks two
builders what the net is fed and what it is scored against -- so wrapping a model that forecasts
something else needs no change to any of it.

The one override is :meth:`SeqVaeLagAttnFsTask._build_raw_target`, and it is one line of work:
the target is the *concatenated* target feature stream rather than the raw signal. The
concatenation happens here rather than in the net for two reasons that are each sufficient. The
net may not know what the two stored blocks are called -- ``nets/`` is checked for exactly those
names -- and the diagnostic plotting callback reaches the target through this same method, so a
model whose target is not the raw trace still gets a figure drawn against the tensor its loss was
computed on.

Beside it sits :attr:`SeqVaeLagAttnFsTask.forecast_rows`, which overrides nothing: the shared
plotting callback resolves it with ``getattr(..., None)`` and the shared page builder turns a
``None`` back into its own raw rows, so naming one here is how a model in another target domain
supplies the two rows of the seven that depend on the domain. It is a *property* rather than a
class attribute because a plain function assigned to a class becomes a bound method, which would
hand the row builder ``self`` as its first argument, and because the two channel facts the page
needs -- the surviving-channel index and the block boundary -- live on the net and can only be
read once there is an instance.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Tuple

import torch

from teb_vae.lag_attn_fs.sample_page import feature_forecast_rows
from teb_vae.lag_attn_rws.sample_page import ForecastRowInputs
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask


class SeqVaeLagAttnFsTask(SeqVaeLagAttnRwsTask):
    r"""Lightning task for :class:`~teb_vae.lag_attn_fs.nets.model.SeqVaeLagAttnFs`.

    Adds exactly one method. The absence of everything else is deliberate and load-bearing: no
    ``training_step`` (the inherited one runs the config-gated loss-spike breaker), no
    ``compute_loss_and_metrics`` (which is where the permutation control, the ``main_loss`` name
    the breaker watches, and the latent-gap diagnostic live), and no constructor -- a second
    keyword schema for the same objective could only drift from the first.

    ``_mu_gap_rms`` needs no override either, which is worth stating because it looks
    target-shaped: it reads the geometry, the coverage floor, the two masks and the two latent
    means, and every one of those is domain-neutral.
    """

    @property
    def forecast_rows(self) -> Callable[[ForecastRowInputs], None]:
        r"""The page's first two rows, bound to this net's channel facts.

        Read off the task by the shared plotting callback and handed to the shared page builder,
        which draws rows $3$ to $7$ itself. Nothing is overridden: the callback resolves this name
        with ``getattr(..., None)`` and the builder reads a ``None`` as "use the raw page", so the
        two raw-target siblings are unaffected by its existing at all.

        The two bound values are what the page cannot recover from the arrays it is handed. The
        keep-index says which declared channel each of the decoder's outputs *is*, which is needed
        both to gather the truth a lane is judged against and to put a channel number on the axis
        that still means the same thing at another reach budget. The block split is where the two
        stored blocks meet on that channel axis, the same boundary ``pred_gap_st`` and
        ``pred_gap_ph`` are reported either side of.

        Returns:
            A callable taking one
            :class:`~teb_vae.lag_attn_rws.sample_page.ForecastRowInputs` and drawing into it.
        """
        gate = self.orig_model.target_gate
        return partial(
            feature_forecast_rows,
            keep_index=None if gate is None else gate.keep_index,
            block_split=int(self.orig_model.TARGET_BLOCK_SPLIT),
        )

    def _build_raw_target(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(target_features, weight)``: the forecast target and its validity signal.

        The target is the two stored target blocks concatenated along the channel axis, in the
        declared order -- $(B, T, c_y)$ on the decimated grid, exactly the stream the net's
        forward is fed. Which of those $c_y$ channels are actually forecast is the *net's*
        decision, made from its target gate's keep-index; handing the whole declared stream over
        is what keeps that index positional into a fixed channel order rather than into whatever
        subset the task happened to pass.

        Reusing :meth:`_build_target_streams` rather than reading the batch again is what makes
        the width check load-bearing here: it compares the two blocks' joint width against the
        model's $c_y$ and names the config key that fixes a mismatch, and the target must be
        checked by the same rule as the input because it *is* the same tensor.

        No delay is applied, and no gather. The delay belongs to the input gate, where it pushes
        each channel's forward reach behind the anchor's causal endpoint; applied to the target it
        would silently ask anchor $t$ to forecast the future of anchor $t - \delta_c$, per
        channel, with every shape downstream unchanged.

        The block boundary is checked here for the same reason the joint width is: the net splits
        its forecast gap at ``TARGET_BLOCK_SPLIT``, a declared number it cannot derive, and this is
        the only place that sees the two blocks separately and can say whether the number is true
        of the data. Nothing else depends on it, so a wrong value would mislabel two reported
        columns and break nothing -- which is exactly the kind of error that survives a run.

        Args:
            batch: A batch from the data module.

        Returns:
            The target stream ``(B, T, c_y)`` and the validity weight ``(B, T)``.

        Raises:
            RuntimeError: If ``weight`` is absent, naming the config key that fixes it; if the two
                blocks' joint width disagrees with the model's ``c_y``; or if the first block's own
                width disagrees with the split the net reports its per-block gaps against.
        """
        y_st, y_ph = self._build_target_streams(batch)
        split = int(self.orig_model.TARGET_BLOCK_SPLIT)
        if int(y_st.shape[-1]) != split:
            raise RuntimeError(
                f"the first target block is {int(y_st.shape[-1])} channels but the model splits "
                f"its per-block forecast gap at {split}. The split reaches no loss term and no "
                f"shape, so leaving it stale would mislabel `pred_gap_st` and `pred_gap_ph` and "
                f"nothing else would fail. Set SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT to this shard's "
                f"own first-block width."
            )
        weight = getattr(batch, "weight", None)
        if weight is None:
            raise RuntimeError(
                "batch has no `weight` field. The decimated weight is the only trustworthy "
                "validity signal for the forecast target -- the stored coefficients carry no "
                "detectable gap sentinel of their own; add 'weight' to "
                "dataset_kwargs.load_fields."
            )
        return torch.cat([y_st, y_ph], dim=-1), weight
