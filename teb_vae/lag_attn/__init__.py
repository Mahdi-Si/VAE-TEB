"""The lag-attention sequential VAE.

A VAE over decimated scattering features that forecasts future FHR structure from its own history
plus lagged uterine-pressure context, selected by a lag-cross-attention over a $L = \\mathrm{max\\_lag} + 1$
window. Transfer-entropy-like readouts fall out of the per-lag KL decomposition.

Layout::

    nets/       the network, framework-free: torch + stdlib + entmax only
    tests/      fast hermetic pytest
    eval/       slow analysis against a checkpoint, run by a human

Variants are configs, not subclasses.

``DESIGN.md`` is this model's contract; ``eval/EVAL.md`` is the evaluation pipeline's, and
``eval/FIGURE_GUIDE.md`` says how to read what a run emits.
"""
