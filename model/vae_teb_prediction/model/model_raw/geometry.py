r"""Config-driven raw/low-rate geometry for the raw-signal VAE-TEB v4 model.

Overview
--------
The raw model ingests a $22$-minute FHR/UP segment sampled at $f_s = 4$ Hz
($L_{\mathrm{raw}} = 5280$ samples) and produces a low-rate token grid of $T = 300$ steps
($0.25$ Hz, $= 20$ minutes) after the learned front end. The mapping is

$$
\underbrace{L_{\mathrm{raw}} = 5280}_{22\ \mathrm{min}}
\;\xrightarrow[\text{stride } D=16]{\text{causal front end}}\;
\underbrace{\tilde T = 330}_{22\ \mathrm{min}}
\;\xrightarrow[\text{crop } [\mathrm{CROP}:\tilde T-\mathrm{CROP})]{}\;
\underbrace{T = 300}_{20\ \mathrm{min}} .
$$

Why $22 \to 20$ minutes, and why crop *after* the transform (edge-zone rationale)
--------------------------------------------------------------------------------
The two one-minute edge zones are load-bearing and have **distinct** purposes:

- **Left $\mathrm{CROP}=15$ tokens = real causal left-context.** The front end is strictly
  causal (left-only padding), so token $t'$ depends only on raw samples $\le n_{t'}$. Feeding
  the extra left minute means the useful region's early tokens see *real* history in their
  receptive field rather than zero-padding, so the useful $20$ minutes are edge-clean. This
  mirrors v3's "compute the scattering transform on the full $22$ min, then trim $1$ min each
  side" edge-effect removal -- we likewise trim **after** the transform (in token space),
  which is why the loader keeps ``trim_minutes: null`` (untrimmed $5280$/$330$). Trimming to
  $20$ min *before* the front end would re-introduce the very edge effect the trim removes.
- **Right $\mathrm{CROP}=15$ tokens = forecast-target headroom.** The last valid anchors
  forecast $H = 30$ low-rate steps ($=2$ min) into the future; those raw targets live in the
  right edge zone (raw samples up to $5039 < 5280$).

The crop-offset (the single most error-prone point)
---------------------------------------------------
A **cropped** anchor $t \in [0, T)$ is the **uncropped** token $t' = t + \mathrm{CROP}$, so its
raw causal endpoint is

$$
n_{\mathrm{raw}}(t) = D\,(t + \mathrm{CROP} + 1) - 1 = 16\,(t + 16) - 1 \quad(\text{NOT } 16(t+1)-1),
$$

and its future forecast block starts **one sample later**, at

$$
\mathrm{future\_block\_start}(t) = n_{\mathrm{raw}}(t) + 1 = D\,(t + \mathrm{CROP} + 1)
= 16\,(t + 16) \quad(=256 \text{ at } t=0),
$$

which is **not** the anchor's own-present start $D\,(t + \mathrm{CROP}) = 16(t + 15)$ ($=240$ at
$t=0$; used only for the anchor-validity mask), and **not** $D(t+1)$.

Everything here is derived from ``raw_len`` and ``decimation`` (never hardcoded to $5280$), so a
dataset built with a different ``len_signal`` fails loudly via :func:`derive_geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Primitive constants (production defaults; all geometry is derived from these).
# ---------------------------------------------------------------------------
RAW_LEN: int = 5280          #: raw samples per segment ($22$ min @ $4$ Hz)
D: int = 16                  #: decimation factor / front-end total stride
CROP: int = 15               #: tokens trimmed each side ($=1$ min @ $0.25$ Hz)
H: int = 30                  #: forecast horizon in low-rate steps ($=2$ min)
WARMUP: int = 30             #: warm-up steps excluded from all losses / TE
MAX_LAG: int = 90            #: max lag index ($\approx 6$ min UP history)
D_MODEL: int = 128           #: internal model width
D_Z: int = 24                #: latent dimension
NUM_HEADS: int = 4           #: attention heads
D_HEAD: int = 32             #: per-head width

# ---------------------------------------------------------------------------
# Derived constants (module-level convenience for the production geometry).
# ---------------------------------------------------------------------------
T_TILDE: int = RAW_LEN // D          #: pre-crop low-rate tokens ($=330$)
T: int = T_TILDE - 2 * CROP          #: final low-rate steps ($=300$, $20$ min)
R: int = D                           #: raw samples per low-rate step ($=16$)


@dataclass(frozen=True)
class RawGeometry:
    r"""A validated, config-driven raw/low-rate geometry.

    All quantities derive from ``raw_len`` and ``decimation``; the remaining fields default to
    the production values. Construct via :func:`derive_geometry`, which validates the
    divisibility and crop-offset identities.

    Attributes:
        raw_len: Raw samples per segment $L_{\mathrm{raw}}$.
        decimation: Front-end total stride $D$ (also the raw substeps per low-rate step $R$).
        crop: Tokens trimmed each side $\mathrm{CROP}$.
        horizon: Forecast horizon $H$ in low-rate steps.
        warmup: Warm-up steps $w$ excluded from losses/TE.
    """

    raw_len: int = RAW_LEN
    decimation: int = D
    crop: int = CROP
    horizon: int = H
    warmup: int = WARMUP

    # -- Derived scalars ----------------------------------------------------
    @property
    def t_tilde(self) -> int:
        r"""Pre-crop token count $\tilde T = L_{\mathrm{raw}} / D$."""
        return self.raw_len // self.decimation

    @property
    def t(self) -> int:
        r"""Final low-rate step count $T = \tilde T - 2\,\mathrm{CROP}$."""
        return self.t_tilde - 2 * self.crop

    @property
    def r(self) -> int:
        r"""Raw substeps per low-rate step $R = D$."""
        return self.decimation

    @property
    def t_valid(self) -> int:
        r"""Number of trained anchors $T_{\mathrm{valid}} = T - H$."""
        return self.t - self.horizon

    # -- Index helpers ------------------------------------------------------
    def token_endpoint_uncropped(self, t_prime: int) -> int:
        r"""Raw causal endpoint of the **uncropped** token $t'$: $D(t'+1) - 1$.

        Args:
            t_prime: An uncropped token index in $[0, \tilde T)$.

        Returns:
            The largest raw index token $t'$ may causally depend on.
        """
        return self.decimation * (t_prime + 1) - 1

    def n_raw(self, t: int) -> int:
        r"""Raw causal endpoint of the **cropped** anchor $t$: $D(t + \mathrm{CROP} + 1) - 1$.

        Args:
            t: A cropped anchor index in $[0, T)$.

        Returns:
            The largest raw index anchor $t$ may causally depend on.
        """
        return self.decimation * (t + self.crop + 1) - 1

    def own_present_start(self, t: int) -> int:
        r"""Start of the anchor's own present block: $D(t + \mathrm{CROP})$.

        This is what the anchor-validity mask $m^{\mathrm{low}}_t$ reads; it is **distinct**
        from :meth:`future_block_start`.
        """
        return self.decimation * (t + self.crop)

    def future_block_start(self, t: int) -> int:
        r"""First raw sample of the forecast block: $n_{\mathrm{raw}}(t) + 1 = D(t+\mathrm{CROP}+1)$.

        Args:
            t: A cropped anchor index in $[0, T)$.

        Returns:
            The raw index at which anchor $t$'s $2$-minute future forecast begins.
        """
        return self.n_raw(t) + 1

    def future_block_indices(self, t: int) -> List[List[int]]:
        r"""The $(H, R)$ grid of raw target indices for cropped anchor $t$.

        Index $[\tau, r]$ (with $\tau \in [0, H)$, $r \in [0, R)$) is
        $\mathrm{future\_block\_start}(t) + D\,\tau + r$.

        Args:
            t: A cropped anchor index in $[0, T)$.

        Returns:
            A nested list of shape $(H, R)$ of raw sample indices.
        """
        start = self.future_block_start(t)
        return [
            [start + self.decimation * tau + r for r in range(self.r)]
            for tau in range(self.horizon)
        ]

    def valid_anchor_range(self) -> range:
        r"""The trained-anchor range $[w, T - H)$ (a fully observed $2$-min future exists)."""
        return range(self.warmup, self.t - self.horizon)


def derive_geometry(
    raw_len: int,
    decimation: int,
    *,
    crop: int = CROP,
    horizon: int = H,
    warmup: int = WARMUP,
) -> RawGeometry:
    r"""Build and validate a :class:`RawGeometry` from ``raw_len`` and ``decimation``.

    Asserts the divisibility ($L_{\mathrm{raw}} \bmod D = 0$), positivity ($T > 0$,
    $T_{\mathrm{valid}} > 0$), the crop-offset identity
    ($\mathrm{future\_block\_start}(0) = D(\mathrm{CROP}+1) = n_{\mathrm{raw}}(0) + 1$, distinct
    from the own-present start $D\cdot\mathrm{CROP}$), and that the last valid anchor's forecast
    lands strictly inside the loaded window.

    Args:
        raw_len: Raw samples per segment $L_{\mathrm{raw}}$.
        decimation: Front-end total stride $D$.
        crop: Tokens trimmed each side.
        horizon: Forecast horizon in low-rate steps.
        warmup: Warm-up steps excluded from losses/TE.

    Returns:
        A validated :class:`RawGeometry`.

    Raises:
        ValueError: If ``raw_len`` is not divisible by ``decimation`` or the derived geometry
            is degenerate (non-positive $T$ / $T_{\mathrm{valid}}$).
        AssertionError: If a crop-offset or last-anchor identity fails.
    """
    if decimation <= 0:
        raise ValueError(f"decimation must be positive, got {decimation}")
    if raw_len % decimation != 0:
        raise ValueError(
            f"raw_len ({raw_len}) must be divisible by decimation ({decimation})"
        )

    geo = RawGeometry(
        raw_len=raw_len,
        decimation=decimation,
        crop=crop,
        horizon=horizon,
        warmup=warmup,
    )

    if geo.t <= 0:
        raise ValueError(
            f"non-positive T={geo.t} (t_tilde={geo.t_tilde}, crop={crop}); "
            "crop is too large for this raw_len/decimation"
        )
    if geo.t_valid <= 0:
        raise ValueError(
            f"non-positive T_valid={geo.t_valid} (T={geo.t}, horizon={horizon})"
        )
    if not (0 <= warmup < geo.t - horizon):
        raise ValueError(
            f"warmup ({warmup}) must satisfy 0 <= warmup < T - H ({geo.t - horizon})"
        )

    # Crop-offset identities (the single most error-prone point).
    assert geo.future_block_start(0) == decimation * (crop + 1), (
        "future_block_start(0) must equal D*(CROP+1)"
    )
    assert geo.future_block_start(0) == geo.n_raw(0) + 1, (
        "future_block_start(t) must be n_raw(t) + 1"
    )
    assert geo.own_present_start(0) == decimation * crop, (
        "own_present_start(0) must equal D*CROP"
    )
    assert geo.own_present_start(0) != geo.future_block_start(0), (
        "own-present start and forecast start must be distinct"
    )

    # Last valid anchor's forecast must land strictly inside the loaded raw window.
    last_anchor = geo.valid_anchor_range()[-1]
    last_target_end = geo.future_block_start(last_anchor) + horizon * geo.r - 1
    assert last_target_end < raw_len, (
        f"last-anchor forecast end {last_target_end} must be < raw_len {raw_len}"
    )

    return geo


#: The production geometry (validated at import time).
GEOMETRY: RawGeometry = derive_geometry(RAW_LEN, D)


# ---------------------------------------------------------------------------
# Module-level functional helpers over the production geometry.
# ---------------------------------------------------------------------------
def n_raw(t: int) -> int:
    r"""Raw causal endpoint of cropped anchor $t$ for the production geometry."""
    return GEOMETRY.n_raw(t)


def future_block_start(t: int) -> int:
    r"""First raw target sample of cropped anchor $t$ for the production geometry."""
    return GEOMETRY.future_block_start(t)


def future_block_indices(t: int) -> List[List[int]]:
    r"""The $(H, R)$ raw-target index grid of cropped anchor $t$ for the production geometry."""
    return GEOMETRY.future_block_indices(t)


def valid_anchor_range() -> range:
    r"""The trained-anchor range $[w, T-H) = [30, 270)$ for the production geometry."""
    return GEOMETRY.valid_anchor_range()


def token_endpoint_uncropped(t_prime: int) -> int:
    r"""Raw causal endpoint of the uncropped token $t'$ for the production geometry."""
    return GEOMETRY.token_endpoint_uncropped(t_prime)
