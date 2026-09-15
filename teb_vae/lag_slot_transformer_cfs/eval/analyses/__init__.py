r"""This cell's own analyses, registered on its binding beside the family's shared ones.

Every module here is an analysis under the family's protocol -- see
:mod:`teb_vae.lag_attn_cfs.eval.analyses` for the signature, the four required keys and the
layering rule -- and every one reads the collection this cell's own pass wrote rather than the
model. Six of them, in three kinds:

* **The readouts only this architecture has**, drawn from the results block the pass assembled:
  the scored arms and their paired margins (``arms``), the band suppression and the per-lag
  profile (``lag_suppression``), and the horizon- and block-resolved margins (``resolved_axes``).
  Each writes its own table beside its figures, so a directory of this cell is read down the same
  layout as a lag-attentive cell's: one subdirectory per question.
* **The three that draw this model's own forward**, under the same three names the lag-attentive
  cells register theirs: the per-sample pages (``samples``), the per-recording traces
  (``recording_traces``) and the Captum attributions (``attribution``). Same names, so two run
  directories are read down one layout; this cell's own implementations, because each draws a
  tensor the other architecture does not emit and the other draws one this architecture does not.

What is deliberately **not** here: any analysis that reads an attention distribution or a
per-lag allocation of the divergence. The seven the family has are recorded in the summary as
absent, with the tensor each would have needed, by
:data:`~teb_vae.lag_slot_transformer_cfs.eval.binding.ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE`.
"""
