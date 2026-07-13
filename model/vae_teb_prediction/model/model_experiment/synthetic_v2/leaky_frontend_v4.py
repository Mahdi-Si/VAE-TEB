r"""S4-T02: the leaky (time-pooling, non-causal) front end -- the G0-in-front-end negative control.

`synthetic_v4`'s headline question mirrors v3's G0: does the raw model with a **causal** front end
recover the known TE, and does deliberately breaking front-end causality inflate the null? This
module builds the negative control **without editing `model_raw`**:

* :class:`LeakyRawFrontend` subclasses :class:`CausalRawFrontend`, calls ``super().__init__()`` (so
  the frontend self-guard passes on the causal norms it just built), then **replaces** each block's
  causal per-sample norm with a time-pooling :class:`torch.nn.GroupNorm`. ``GroupNorm`` pools its
  statistics over the whole time axis (and channels) per sample, so the normalised value at token
  $t$ depends on tokens $t' > t$ -- a genuine leak of the future into the past. It is one of the
  ``_FORBIDDEN_NORMS`` the model's ``assert_no_time_pooling_norm`` guard catches, so a standalone
  leaky front end **fails** that guard (proving the control is real).
* :class:`LeakyRawFrontendSeqVaeRawV4` subclasses :class:`SeqVaeRawV4`, calls ``super().__init__()``
  (whose model-level guard on ``frontend_y``/``frontend_u`` passes on the freshly-built causal front
  ends), then **replaces** ``self.frontend_y``/``self.frontend_u`` with :class:`LeakyRawFrontend`
  instances -- post-init, so no guard re-fires and no ``model_raw`` code is touched or bypassed.

:func:`frontend_is_causal` is the probe that certifies the control: it perturbs the raw input
strictly **after** an early token's causal endpoint and checks whether that token's output changed.
It returns ``True`` for :class:`CausalRawFrontend` and ``False`` for :class:`LeakyRawFrontend`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    CausalRawFrontend,
    SeqVaeRawV4,
)

#: Front-end constructor keys that :class:`SeqVaeRawV4` pops before building the causal front ends;
#: mirrored here so the leaky rebuild passes the same ``**fe`` the model used.
_POPPED_FRONTEND_KEYS = ("decoder_head", "basis_size", "sentinel")


def _make_frontend_leaky(frontend: CausalRawFrontend, *, num_groups: int = 8) -> None:
    r"""Replace every block's causal norm with a time-pooling :class:`torch.nn.GroupNorm` in place.

    ``GroupNorm`` normalises over $(C, T)$ within each group per sample, so it pools statistics over
    the full time axis -- non-causal, and one of the ``_FORBIDDEN_NORMS`` the model guard rejects.

    Args:
        frontend: A built :class:`CausalRawFrontend` whose ``blocks[i].norm`` are swapped.
        num_groups: Requested group count; falls back to $1$ when it does not divide a block's
            channel count.
    """
    for block in frontend.blocks:
        c = int(block.refine.out_channels)
        g = num_groups if (num_groups > 0 and c % num_groups == 0) else 1
        block.norm = nn.GroupNorm(num_groups=g, num_channels=c)


class LeakyRawFrontend(CausalRawFrontend):
    r"""A :class:`CausalRawFrontend` with each block's norm swapped for a time-pooling GroupNorm.

    Builds identically to the causal front end (so ``super().__init__`` passes the frontend
    self-guard), then makes it non-causal post-construction. Kept a distinct class so the causality
    probe / guard test can target it directly.
    """

    def __init__(self, *, leaky_num_groups: int = 8, **kwargs) -> None:
        r"""Build a causal front end, then make it leaky.

        Args:
            leaky_num_groups: Group count for the replacement :class:`torch.nn.GroupNorm`.
            **kwargs: Forwarded verbatim to :class:`CausalRawFrontend`.
        """
        super().__init__(**kwargs)  # builds causal norms; the frontend self-guard passes here
        _make_frontend_leaky(self, num_groups=leaky_num_groups)


class LeakyRawFrontendSeqVaeRawV4(SeqVaeRawV4):
    r"""A :class:`SeqVaeRawV4` whose front ends are :class:`LeakyRawFrontend` (the ``frontend_noncausal`` arm).

    ``super().__init__`` builds the causal front ends and runs the model-level
    ``assert_no_time_pooling_norm`` on them (passes); this subclass then **replaces**
    ``self.frontend_y``/``self.frontend_u`` with leaky front ends built from the same constructor
    arguments -- so no guard re-fires and no ``model_raw`` code is edited or its guard bypassed. The
    replacement front ends receive :class:`CausalRawFrontend`'s default weight init (the model's own
    post-build init pass does not re-run on them); acceptable for a negative control.
    """

    def __init__(self, *, leaky_num_groups: int = 8, **kwargs) -> None:
        r"""Build the causal model, then swap in leaky front ends.

        Args:
            leaky_num_groups: Group count for the replacement GroupNorm in each front end.
            **kwargs: Forwarded verbatim to :class:`SeqVaeRawV4` (``frontend``, ``raw_len``,
                ``decimation``, ``fhr_mean``/``fhr_std``/``up_mean``/``up_std``, ``**v3_kwargs``).
        """
        super().__init__(**kwargs)

        # Mirror SeqVaeRawV4.__init__'s front-end construction (vae_teb_raw_v4.py:392-409): the same
        # ``fe`` (frontend dict minus the decoder-head keys and sentinel), stream, fixed z-score
        # stats, and geometry -- only the norm differs (leaky).
        fe = dict(kwargs.get("frontend") or {})
        for key in _POPPED_FRONTEND_KEYS:
            fe.pop(key, None)
        raw_len = int(kwargs.get("raw_len", 5280))
        decimation = int(kwargs.get("decimation", 16))
        self.frontend_y = LeakyRawFrontend(
            stream="y", mean=float(kwargs.get("fhr_mean", 0.0)),
            std=float(kwargs.get("fhr_std", 1.0)), raw_len=raw_len, decimation=decimation,
            sentinel=None, leaky_num_groups=leaky_num_groups, **fe,
        )
        self.frontend_u = LeakyRawFrontend(
            stream="u", mean=float(kwargs.get("up_mean", 0.0)),
            std=float(kwargs.get("up_std", 1.0)), raw_len=raw_len, decimation=decimation,
            sentinel=None, leaky_num_groups=leaky_num_groups, **fe,
        )


def frontend_is_causal(
    frontend: CausalRawFrontend, *, seed: int = 0, atol: float = 1e-5, token: int = 5,
) -> bool:
    r"""Probe whether a front end is causal: does a future raw sample leak into an earlier token?

    Runs the front end on a random raw input, then re-runs with the raw signal perturbed **strictly
    after** the causal endpoint $n_{\mathrm{raw}}(t_0) = R\,(t_0 + \mathrm{crop} + 1) - 1$ of an
    early output token $t_0$, and checks whether token $t_0$'s output changed. A causal front end
    leaves it bit-identical (returns ``True``); a leaky (time-pooling) front end shifts it (returns
    ``False``).

    Args:
        frontend: A built :class:`CausalRawFrontend` (or subclass).
        seed: RNG seed for the deterministic probe.
        atol: Absolute tolerance for the "unchanged" comparison.
        token: The early output-token index $t_0$ to test (clamped to the valid range).

    Returns:
        ``True`` iff perturbing the raw signal after $t_0$'s causal endpoint leaves token $t_0$
        unchanged.
    """
    was_training = frontend.training
    frontend.eval()
    try:
        g = torch.Generator().manual_seed(int(seed))
        length = int(frontend.raw_len)
        raw = torch.randn(1, length, generator=g)
        mask = torch.ones(1, length)
        with torch.no_grad():
            out0 = frontend(raw, mask)  # (1, T, d_raw)
        t0 = max(0, min(int(token), int(out0.shape[1]) - 1))
        # Causal raw endpoint of output token t0 (== cropped anchor t0): last raw sample it may see.
        n_end = int(frontend.decimation) * (t0 + int(frontend.crop) + 1) - 1
        raw2 = raw.clone()
        n_future = length - (n_end + 1)
        if n_future <= 0:
            return True  # no future samples to perturb -> trivially unchanged
        # Perturb strongly so a leak is unmistakable (well above atol).
        raw2[:, n_end + 1:] = 10.0 * torch.randn(1, n_future, generator=g)
        with torch.no_grad():
            out1 = frontend(raw2, mask)
        return bool(torch.allclose(out0[:, t0], out1[:, t0], atol=atol))
    finally:
        if was_training:
            frontend.train()
