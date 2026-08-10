"""Network components for the feature-domain lag-attention VAE.

Every module here imports only ``torch``, the standard library, ``entmax`` and the framework-free
``teb_vae`` net layers. No Lightning, no config, no logging, no ``torch.distributed``, and no
batch field names -- these are nets, and what feeds them is the caller's problem.

The last of those is sharper here than in either sibling. This model's target is built from two
*named stored blocks*, and naming them inside ``nets/`` -- in code or in a docstring -- is exactly
what the framework-free guard forbids. So the concatenation of the two blocks happens task-side,
where batch field names belong, and what arrives here is one ``(B, T, c_y)`` tensor whose origin
this layer does not know.

What is written here is the one thing this model owns, split across two modules for the one reason
that matters: ``feature_target.py`` holds the *target domain* -- the decoder width follows the
target gate's surviving-channel count rather than the raw samples per step, and the objective is
handed a gathered feature block instead of a gathered raw window -- and ``model.py`` holds the
*composition*, which is that target domain plus an encoder model. The split is what lets a second
forecaster reach the same target through different encoders without a second copy of the target.
Everything else -- both encoders, the channel gate, the prior and posterior heads, the lag
cross-attention, the decoder core, the paired reparameterisation and the seven-term objective -- is
imported from ``teb_vae.lag_attn_rws.nets`` and ``teb_vae.lag_attn.nets``.
"""
