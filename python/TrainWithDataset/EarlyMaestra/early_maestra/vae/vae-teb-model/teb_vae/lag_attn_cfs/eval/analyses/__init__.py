r"""The analyses, and the one protocol every one of them obeys.

An analysis is a function

.. code-block:: python

    def run_<name>_analysis(context, *, eval_config, output_dir, probe) -> Dict[str, Any]

and that signature is fixed, checked by ``tests/test_eval_protocol.py`` by inspection rather than
by convention. The runner calls every analysis the same way, so registering one is a line in the
registry rather than a branch in the run; and an analysis whose signature drifts is caught at the
gate rather than at the ninth step of a multi-hour pass.

``probe`` is the **population record** the collection pass produced -- ``loader_probe.json``'s
content -- and is neither a model probe nor :mod:`teb_vae.lag_attn_cfs.eval.probe`. The runner
passes it as ``probe_record``; an analysis that does not need the population ignores it.

**An analysis does not touch the model.** The decoder pass over four scored latent branches at $K$
draws, plus the fifth KL-only ``kld_source_null`` arm, happens once in the shared collection pass,
and what it leaves behind -- two durable tables, a vector sidecar, the retained heavy arrays, and
the readouts themselves -- is what an analysis reads, through :class:`AnalysisContext`. That is
what makes ``--only <anything>`` against a finished run directory work with no checkpoint loaded
and no GPU, which is the whole point of splitting collection from emission. It is also why the
layering test forbids ``analyses/*`` from importing Lightning, ``model.*``, or this package's
``task`` / ``trainer`` / ``sample_page``.

One consequence of that rule is load-bearing in this target domain and is easy to trip over. The
per-channel readouts are indexed on the **98 kept** target channels while the channel-to-band map
is over the **102 declared** ones, and ``analyses/*`` may not ask ``model.target_gate`` which is
which. So the kept-axis map is *persisted* -- the collection pass records ``target_keep_index``,
``band_partition`` emits the kept-axis channel map, and ``spectral_skill`` reads that CSV off disk
-- exactly the file-on-disk dependency ``cross_subgroup`` already has on the per-recording tables
above it.

**No analysis imports another.** Anything two of them share moves one layer down -- into
``metrics``, ``events`` or the reuse seam -- rather than being reached for sideways. The layering
test reports a sideways import whether it is written absolutely, relatively, or lazily inside the
one function that needs it.

**Every analysis returns the same four keys**, because the runner reads them without knowing what
the analysis did:

* ``n_samples`` -- how many segments this analysis actually scored, or ``None`` when it scored
  none because it describes something other than a population (the target channel map is the
  standing example). The coverage block compares this across uncapped analyses, and two analyses
  reporting different populations is a finding: metrics from them do not describe the same
  recordings, and nothing else in the output would say so.
* ``composition`` -- how those samples split across the cohorts the clinical questions are asked
  in, or ``{}``.
* ``plan`` -- what the analysis retained or capped, carrying at least ``capped``. A capped
  analysis is excluded from the population comparison above rather than reported as a
  disagreement.
* ``grouped_frames`` -- optional. Each entry *names a per-sample CSV on disk* plus the columns to
  resolve by group; the runner fans the by-class and by-subgroup variants over them. The frames
  are named rather than returned because the return value is serialised into ``summary.json``,
  and because the fan-out is deliberately the runner's job: an analysis that had to remember to
  emit its own grouped variants is an analysis added later that will not.
  ``*_by_clinical_class.pdf`` and ``*_by_subgroup.pdf`` are reserved filenames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

#: Keys every analysis's return value carries. ``grouped_frames`` is optional and is therefore not
#: on this list -- an analysis with no per-sample frame has no grouped variant to declare.
REQUIRED_RESULT_KEYS = ("n_samples", "composition", "plan")

#: The optional key the runner fans grouped variants over. Named here so the runner and the
#: protocol test agree on the spelling.
GROUPED_FRAMES_KEY = "grouped_frames"


@dataclass(frozen=True)
class AnalysisContext:
    """Everything an analysis is given about the run, other than its own arguments.

    ``collection`` is what almost every analysis reads and is the whole point of the collect/emit
    split: the expensive pass has already happened, so ``--only <anything>`` against a finished
    directory works with no checkpoint loaded and no GPU.

    ``task`` and ``loader`` are the deliberate exception, and exactly one analysis takes them: the
    per-sample diagnostic pages. A page is the *whole* forward output of one segment and the
    extreme-value pages are chosen by sorting a table the pass had not written when it ran, so
    neither can be served by retention. Both are ``None`` on a pass with no checkpoint, and the
    analysis that wants them records a skip rather than assuming them; every other analysis reads
    the tables, and the layering test still forbids ``analyses/*`` from *importing* Lightning,
    ``model.*`` or this package's ``task`` / ``trainer`` / ``sample_page``.

    Attributes:
        collection: What the shared pass produced -- the per-sample and per-anchor tables, the
            per-sample vector readouts, the retained heavy arrays, the readouts and the
            provenance record. Typed loosely so this module stays free of an import that would
            pull ``torch`` into the protocol definition.
        config: The merged run configuration, for what describes the *data* rather than the
            model: which shards were evaluated, and under which dataset settings.
        task: The loaded task, or ``None`` when this pass built no model.
        loader: The evaluation dataloader, or ``None`` when this pass built none -- which is the
            case whenever the tables were read back rather than collected.
    """

    collection: Any
    config: Dict[str, Any] = field(default_factory=dict)
    task: Any = None
    loader: Any = None


__all__ = ["AnalysisContext", "GROUPED_FRAMES_KEY", "REQUIRED_RESULT_KEYS"]
