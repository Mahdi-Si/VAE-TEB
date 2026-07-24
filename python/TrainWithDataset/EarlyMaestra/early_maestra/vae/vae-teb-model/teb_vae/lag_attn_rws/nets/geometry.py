r"""Anchor-to-raw index geometry for the **trimmed** loader grid.

The loader (``trim_minutes: 1.0``) applies the symmetric crop *before* the model sees anything:
raw signals arrive as ``data[240:-240]`` ($L_{\mathrm{raw}} = 4800$) and the decimated feature
blocks as ``data[:, 15:-15]`` ($T = 300$). Decimated step $t$ therefore covers raw samples
$[D t,\, D(t+1))$ with $D = 16$, so its causal endpoint and the first sample of its forecast
block are

$$
n_{\mathrm{raw}}(t) = D\,(t + 1) - 1 = 16\,(t+1) - 1, \qquad
\mathrm{future\_block\_start}(t) = n_{\mathrm{raw}}(t) + 1 = 16\,(t+1).
$$

**Why this differs from the untrimmed form.** A model that loads untrimmed data
($L_{\mathrm{raw}} = 5280$, $\tilde T = 330$) and crops $\mathrm{CROP} = 15$ tokens per side
*inside* the model maps its cropped anchor $t$ to the uncropped token $t + \mathrm{CROP}$, so its
endpoint is $D\,(t + \mathrm{CROP} + 1) - 1 = 16\,(t + 16) - 1$ and its forecast starts at
$16\,(t + 16)$ ($= 256$ at $t = 0$). Here the loader has already applied the crop, so there is no
$\mathrm{CROP}$ offset anywhere: the forecast of anchor $0$ starts at raw index $16$, not $256$.
Each formula is wrong on the other grid -- off by exactly $\mathrm{CROP} \cdot D = 240$ raw
samples, one minute, and nothing downstream fails loudly on the shift. Because the crop does not
exist on this grid, neither do the ``crop``, ``t_tilde``, ``own_present_start`` or
``token_endpoint_uncropped`` notions of the untrimmed geometry: an anchor's own present block is
simply ``weight[t]``.

The geometry is an explicit attribute of whatever consumes it. There is deliberately no
module-level production singleton for functions to default to: the test geometry is a different
instance, and a silent default is exactly how the wrong grid's formula would creep in.

Run ``python -m teb_vae.lag_attn_rws.nets.geometry`` to print the derived production table.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrimmedRawGeometry:
    r"""A validated anchor-to-raw index map for a trimmed segment.

    All derived quantities follow from the four fields; ``__post_init__`` validates the
    divisibility, positivity and index identities, so an unvalidated instance cannot exist.

    Attributes:
        raw_len: Raw samples per trimmed segment $L_{\mathrm{raw}}$ (production: $4800$).
        decimation: Raw samples per decimated step $D$ (also $R$, the raw samples per horizon
            token; production: $16$).
        horizon: Forecast horizon $H$ in decimated steps (production: $30$).
        warmup: Leading steps $w$ excluded from every loss (production: $30$).
    """

    raw_len: int
    decimation: int
    horizon: int
    warmup: int

    def __post_init__(self) -> None:
        """Validate the geometry, so no invalid instance can escape construction.

        Raises:
            ValueError: On non-positive ``decimation``/``horizon``, a ``raw_len`` that is not a
                multiple of ``decimation``, a horizon that leaves no valid anchor, or a warmup
                outside $[0, T - H)$.
        """
        if self.decimation < 1:
            raise ValueError(f"decimation must be >= 1, got {self.decimation}")
        if self.raw_len % self.decimation != 0:
            raise ValueError(
                f"raw_len ({self.raw_len}) must be divisible by decimation "
                f"({self.decimation}); a trimmed segment holds a whole number of steps"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.t_valid < 1:
            raise ValueError(
                f"degenerate geometry: T={self.t} with horizon={self.horizon} leaves "
                f"T_valid={self.t_valid} anchors with a fully observed forecast window"
            )
        if not 0 <= self.warmup < self.t_valid:
            raise ValueError(
                f"warmup ({self.warmup}) must satisfy 0 <= warmup < T - H "
                f"({self.t_valid}); otherwise no anchor survives the warm-up mask"
            )

        # Index identities of the trimmed grid. These cannot fail from bad *inputs* (the checks
        # above already rejected those); they pin the formulas themselves against the untrimmed
        # variant, whose 16(t+16) form is off by one minute here.
        assert self.future_block_start(0) == self.decimation
        assert self.n_raw(0) == self.decimation - 1
        assert self.future_block_start(0) == self.n_raw(0) + 1
        last_anchor = self.t_valid - 1
        last_target_end = self.future_block_start(last_anchor) + self.horizon * self.r - 1
        assert last_target_end == self.raw_len - 1

    @property
    def t(self) -> int:
        r"""Decimated step count $T = L_{\mathrm{raw}} / D$."""
        return self.raw_len // self.decimation

    @property
    def t_valid(self) -> int:
        r"""Anchors with a fully observed forecast window: $T_{\mathrm{valid}} = T - H$."""
        return self.t - self.horizon

    @property
    def r(self) -> int:
        r"""Raw samples per horizon token $R = D$."""
        return self.decimation

    def n_raw(self, t: int) -> int:
        r"""Raw causal endpoint of anchor $t$: $D\,(t + 1) - 1$.

        Args:
            t: An anchor index in $[0, T)$.

        Returns:
            The largest raw index anchor $t$ may causally depend on.
        """
        return self.decimation * (t + 1) - 1

    def future_block_start(self, t: int) -> int:
        r"""First raw sample of anchor $t$'s forecast block: $n_{\mathrm{raw}}(t) + 1 = D\,(t+1)$.

        Args:
            t: An anchor index in $[0, T)$.

        Returns:
            The raw index at which anchor $t$'s forecast target begins.
        """
        return self.n_raw(t) + 1

    def valid_anchor_range(self) -> range:
        r"""The trained-anchor range $[w, T - H)$."""
        return range(self.warmup, self.t_valid)


if __name__ == "__main__":
    # The production table, for checking the derived indices by eye against the untrimmed
    # variant (model_raw's grid, whose forecast of anchor 0 starts at 256, not 16).
    geometry = TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30)
    first, last = 0, geometry.t_valid - 1
    print("trimmed-grid geometry (raw_len=4800, decimation=16, horizon=30, warmup=30)")
    print(f"  T = {geometry.t}, T_valid = {geometry.t_valid}, R = {geometry.r}")
    print(f"  trained anchors: [{geometry.valid_anchor_range().start}, "
          f"{geometry.valid_anchor_range().stop})")
    for t in (first, last):
        start = geometry.future_block_start(t)
        stop = start + geometry.horizon * geometry.r
        print(f"  anchor {t:>3}: n_raw = {geometry.n_raw(t):>4}, forecast [{start}, {stop})")
    print(f"  contrast: an untrimmed grid with an in-model crop of 15 starts anchor 0's "
          f"forecast at 16*(0+16) = {16 * 16}, one minute later")
