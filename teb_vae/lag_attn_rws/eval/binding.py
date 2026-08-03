r"""The four facts this pipeline cannot derive about the model it is evaluating.

Everything else in this package reads a model through an interface two architectures already
share: the objective, the geometry, the data contract, the prior head, the lag cross-attention and
the head-structured posterior are one implementation, and the analyses reach only into ``nets``
and the tables the collection pass wrote. What is *not* derivable is which class to rebuild, which
constructor keys mean enough to reconcile against a checkpoint, what the model's own encoder has
to disclose about its causal standing, and which committed override delta belongs to it.

Those four facts are what a :class:`ModelBinding` carries, and passing one is what lets a second
architecture reuse this pipeline rather than fork it. A fork is how two models that must stay
comparable stop being comparable: an analysis fixed on one side keeps its bug on the other, and
the two ``summary.json`` files stop being readable side by side long before anyone notices.

**This module names no model class**, deliberately. Naming one would make it layer 1 and cost
every importer a ``torch`` import; the concrete instances therefore live beside the code that
already constructs the model. Its own imports are stdlib only, so an acceptance gate or a
documentation test can read the type without a numeric stack installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple


@dataclass(frozen=True)
class ModelBinding:
    """Which model a run of this pipeline is evaluating, and what only that model knows.

    Frozen because a binding is a declaration rather than a run-time setting: every field decides
    what the numbers in a run *mean*, and a value mutated after the run started would leave a
    ``summary.json`` describing a contract that was not in force when the tables were collected.

    Attributes:
        model_cls: The network class rebuilt from a checkpoint's own ``model_kwargs``, and the
            name every class-mismatch refusal is built from. Wrong, and the run either refuses
            with a message naming the wrong class or -- worse, if the two constructors happen to
            accept the same keys -- evaluates one architecture under another's name.
        task_cls: The objective wrapper the model is scored through. Wrong, and the readouts are
            some other loss reported under this one's column headings.
        tag: The output-directory fallback used when ``general_config.tag`` is absent, as
            ``<tag>-eval``. Wrong, and two models' runs land in one directory and are told apart
            only by timestamp.
        geometry_keys: The constructor keys reconciled against the checkpoint's ``model_kwargs``.
            A key missing here is a key the config may contradict silently; a key here that the
            model does not accept can never match and refuses every run.
        encoder_disclosure: Called with the rebuilt net, returning the encoder-specific half of
            the causality record. What is true of one encoder is not true of the other, and a
            shared key that means nothing in one of them is worse than two honest blocks.
        overrides_path: The committed evaluation override delta merged over a checkpoint's own
            resolved config. Wrong, and the run evaluates the right checkpoint against another
            package's holdout split.
        extra_analyses: Analyses only this model can have, merged onto the shared registry
            **after** it and in declaration order. Empty for a model that adds none, which is the
            common case and says so by omission. A name already in the shared registry is a
            collision rather than an override: silently replacing a shared implementation would
            make two models report different things under one name.
        headline_scalars: Additional ``(name, path into results)`` entries appended to the shared
            headline registry, for what :attr:`extra_analyses` produces. Appended rather than
            merged into the shared tuple, and empty for a model that adds none: the shared
            registry's every path must resolve on a shared run, so a model-specific entry there
            would read as a number every other model failed to produce. A number that stays out of
            the headline stays out of every arm table too, which is why an extra analysis with a
            scalar worth comparing declares it here.
    """

    model_cls: type
    task_cls: type
    tag: str
    geometry_keys: Tuple[str, ...]
    encoder_disclosure: Callable[[Any], Dict[str, Any]]
    overrides_path: Path
    extra_analyses: Mapping[str, Any] = field(default_factory=dict)
    headline_scalars: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
