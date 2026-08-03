r"""The analyses only this architecture can have.

Seventeen of this package's eighteen analyses are the sibling's, imported through the shared
registry rather than copied, and this directory holds the exception:
:mod:`.encoder_attention`, which profiles the encoder self-attention that *is* the replacement.
There is nothing for it to answer in a model whose history encoders are recurrent.

The protocol is the shared one, unchanged and deliberately not restated --
``teb_vae.lag_attn_rws.eval.analyses`` defines it, the runner calls every analysis the same way,
and an analysis registered here returns the same four keys as one registered there. What differs
is only where it is named: a local analysis is registered on ``TRF_BINDING.extra_analyses`` and
merged after the shared registry, so it cannot reach the sibling and cannot reorder the shared
run.

The layering rules carry over too: **no analysis in this package imports another of them**, and
anything two would share moves one layer down -- into this package's own
:mod:`~teb_vae.lag_attn_transformer_rws.eval` modules, or into the shared ``metrics`` / ``frames``
/ ``figures_seam`` seams. The sibling's *eval* package is permitted at every layer, which is the
whole design and includes its analyses; the rule enforced by the layering walk is therefore the
within-package one, and reaching into a sibling analysis would be caught by review rather than by
the test.
"""
