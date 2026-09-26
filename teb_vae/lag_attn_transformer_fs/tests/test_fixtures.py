r"""The two facts about the spliced conftest that the rest of the suite silently relies on.

This conftest takes its constructor keyword sets from the conv-Transformer suite and its batch
machinery and budget resolver from the feature suite. The sibling suites own the pattern, the probes
and the tolerances; what is checked here is the splice.

1. **The two halves agree on the geometry they share.** The imported budget resolver and batch
   builder read $c_y$, $c_u$, the warm-up, $R$ and ``use_up_st`` off the *feature* suite's shipped
   set while every model here is built from the conv-Transformer one.
2. **The tiny guard's delays are non-zero and distinct**, which is what makes every
   never-delayed-target assertion specific: a target built through the gate would be wrong by a
   different number of steps in each channel.
"""
from __future__ import annotations

from teb_vae.lag_attn_fs.tests.conftest import SHIPPED_KWARGS as FEATURE_SHIPPED_KWARGS
from teb_vae.lag_attn_transformer_fs.tests.conftest import SHARED_GEOMETRY_KEYS, SHIPPED_KWARGS


def test_the_shared_geometry_keys_agree_between_the_two_shipped_sets():
    """If the two ever diverged, the resolved keep-index would describe a different stream than the
    one the model declares -- and the gather is positional into the declared width, so it would
    silently take the wrong channels rather than fail."""
    for key in SHARED_GEOMETRY_KEYS:
        assert SHIPPED_KWARGS[key] == FEATURE_SHIPPED_KWARGS[key], key


def test_the_tiny_guards_delays_are_nonzero_and_distinct(tiny_gated):
    delays = tiny_gated["target_delays"]

    assert len(set(delays)) == len(delays)
    assert max(delays) > 0
    assert len(delays) == len(tiny_gated["target_keep_index"])
