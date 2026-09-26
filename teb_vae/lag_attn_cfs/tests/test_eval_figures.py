r"""The properties this package's figure seam must have before any analysis draws through it.

The panels themselves are the shared layer's and are tested there. What is checked here:

**Importing must not restyle anything.** Styling mutates global ``rcParams``, so an import-time
call silently restyles every other figure produced in the same process. Checked in a subprocess,
because this session has imported the module long before the test runs and an in-process check
would pass no matter what.

**The palette is a table rather than an assignment pass**, which is what makes it
order-independent: a cohort asked for alone, among others, or in another order comes back the same
colour, so two figures of overlapping cohorts can be put side by side. Each subgroup is a shade of
its own class's hue, and the subgroups of one class are distinguishable from each other.
"""
from __future__ import annotations

import subprocess
import sys

from teb_vae.lag_attn_cfs.eval import figures_seam
from teb_vae.lag_attn_cfs.eval._reuse import labels

from .conftest import _REPO_ROOT


# =================================================================================================
# The seam
# =================================================================================================
def test_importing_the_seam_does_not_restyle_the_process() -> None:
    """In a subprocess: this session imported the module long ago, so an in-process comparison
    would pass whatever the module does at import time."""
    source = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "before = dict(plt.rcParams)\n"
        "from teb_vae.lag_attn_cfs.eval import figures_seam\n"
        "moved = sorted(k for k, v in plt.rcParams.items() if before.get(k) != v)\n"
        "print(','.join(moved))\n"
        # And the styling is available, so it is opt-in rather than absent.
        "figures_seam.configure_figure_style()\n"
        "assert sorted(k for k, v in plt.rcParams.items() if before.get(k) != v)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"importing figures_seam moved rcParams {completed.stdout.strip()}; styling must be an "
        f"explicit call at run start, not an import side effect"
    )


# =================================================================================================
# The palette
# =================================================================================================
def test_each_clinical_class_carries_its_conventional_hue() -> None:
    """Green, amber, red by severity: the channel test any replacement palette has to satisfy."""
    colours = figures_seam.CLINICAL_CLASS_COLORS

    assert set(colours) == {"healthy", "acidosis", "hie"}
    red, green, blue = (
        {name: int(value[index:index + 2], 16) for name, value in colours.items()}
        for index in (1, 3, 5)
    )
    assert green["healthy"] > red["healthy"] and green["healthy"] > blue["healthy"]
    assert red["hie"] > green["hie"] and red["hie"] > blue["hie"]
    # Amber is red-plus-green with little blue, which is what separates it from both neighbours.
    assert red["acidosis"] > blue["acidosis"] and green["acidosis"] > blue["acidosis"]


def test_every_canonical_subgroup_is_a_shade_of_its_own_class() -> None:
    """The property the eight-cohort figures are readable because of: a violin's hue says which
    class it belongs to before its label is read."""
    for group in labels.CANONICAL_SUBGROUPS:
        shade = figures_seam.SUBGROUP_COLORS[group]
        red, green, blue = (int(shade[index:index + 2], 16) for index in (1, 3, 5))
        if group.startswith("healthy"):
            assert green > red and green > blue, group
        elif group.startswith("acidosis"):
            assert red > blue and green > blue, group
        else:
            assert red > green and red > blue, group


def test_the_subgroups_of_one_class_are_distinguishable_from_each_other() -> None:
    """A shading range that collapsed would give four identical green violins, which is worse than
    four unrelated hues: the figure would read as one cohort drawn four times."""
    healthy = [
        figures_seam.SUBGROUP_COLORS[name]
        for name in labels.CANONICAL_SUBGROUPS
        if name.startswith("healthy")
    ]

    assert len(set(healthy)) == len(healthy) == 4
    # Monotone in luminance across the canonical order, so the shading itself carries the order.
    luminance = [sum(int(value[index:index + 2], 16) for index in (1, 3, 5)) for value in healthy]
    assert luminance == sorted(luminance, reverse=True)


def test_a_cohort_keeps_its_colour_whichever_others_a_figure_contains() -> None:
    """Order-independence is what lets two figures of overlapping cohorts be compared. The shared
    palette assigns colours in arrival order for anything it does not know, and that is exactly the
    failure this table replaces for the eleven labels it does."""
    every = list(figures_seam.CLINICAL_CLASS_COLORS) + list(labels.CANONICAL_SUBGROUPS)
    resolved = figures_seam.group_colors(every)

    assert figures_seam.group_colors(list(reversed(every))) == resolved
    for name in every:
        assert figures_seam.group_colors([name]) == {name: resolved[name]}


def test_an_unknown_cohort_still_receives_a_colour() -> None:
    """A non-canonical shard stem must be drawn, not dropped, so it falls back to the shared
    palette rather than to ``None`` -- which matplotlib would read as "use the default"."""
    resolved = figures_seam.group_colors(["healthy", "not_a_canonical_shard"])

    assert set(resolved) == {"healthy", "not_a_canonical_shard"}
    assert resolved["not_a_canonical_shard"].startswith("#")
