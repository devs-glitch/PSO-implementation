from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from softpy import FloatVectorCandidate

TINY: float = 1e-12
FloatArray = NDArray[np.floating]


def _normalise_bounds(
    lower: FloatArray,
    upper: FloatArray,
    size: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Validate per-dimension box bounds and return float64 copies along with the span.

    - Enforces exact shape (size,) to catch broadcasting mistakes early.
    - Uses independent float64 arrays to avoid hidden aliasing and dtype drift.
    - Requires strictly positive span in every dimension (upper > lower).
    """
    lower_raw = np.asarray(lower)
    upper_raw = np.asarray(upper)

    # Shape must match the dimensionality of the search space.
    if lower_raw.shape != (size,) or upper_raw.shape != (size,):
        raise ValueError(
            f"`lower`/`upper` must have shape ({size},); "
            f"got {lower_raw.shape} and {upper_raw.shape}"
        )

    # Make defensive float64 copies (immutable intent is enforced later).
    lower_arr = np.array(lower_raw, dtype=np.float64, copy=True)
    upper_arr = np.array(upper_raw, dtype=np.float64, copy=True)

    # Positive span is a fundamental invariant for reflection and clipping.
    span = upper_arr - lower_arr
    if np.any(span <= 0):
        bad = np.where(span <= 0)[0]
        raise ValueError(
            f"lower ≥ upper for dimension(s) {bad.tolist()}; "
            "each dimension must have a strictly positive range"
        )
    return lower_arr, upper_arr, span


class ParticleCandidate(FloatVectorCandidate):
    """
    PSO particle compatible with `softpy.FloatVectorCandidate`.

    - `_lower/_upper` are frozen, private copies used by all internal operations.
      This prevents accidental external mutations from violating invariants.
    - Separate copies are passed to the base class (`self.lower/self.upper`)
      purely for interoperability with external/softpy code.
    """

    VMAX_RATIO_DEFAULT: float = 0.4  # Per-dimension velocity cap relative to span.

    def __init__(
        self,
        size: int,
        lower: FloatArray,
        upper: FloatArray,
        position: FloatArray,
        velocity: FloatArray,
        inertia: float,
        wl: float,
        wn: float,
        wg: float,
    ) -> None:
        # Dimension must be a positive integer.
        self.size = int(size)
        if self.size < 1:
            raise ValueError("`size` must be a positive integer (>= 1)")

        # Validate user bounds once and freeze internal copies.
        l_arr, u_arr, _ = _normalise_bounds(lower, upper, self.size)
        self._lower = np.array(l_arr, copy=True); self._lower.setflags(write=False)
        self._upper = np.array(u_arr, copy=True); self._upper.setflags(write=False)

        # Position must match dimensionality; store as base-class buffer.
        pos_arr = np.asarray(position, dtype=float).copy()
        if pos_arr.shape != (self.size,):
            raise ValueError(f"position must have shape ({self.size},); got {pos_arr.shape}")

        # Initialize the base type with independent bound copies (no aliasing).
        super().__init__(self.size, self._lower.copy(), self._upper.copy(), pos_arr)

        # Velocity is particle-local state; shape must match position.
        self.velocity = np.asarray(velocity, dtype=float).copy()
        if self.velocity.shape != (self.size,):
            raise ValueError(f"velocity must have shape ({self.size},); got {self.velocity.shape}")

        # PSO weight sanity checks (convex combination) and inertia range.
        if not np.isclose(wl + wn + wg, 1.0):
            raise ValueError("wl + wn + wg must sum to 1.0")
        if not (0.0 <= wl <= 1.0 and 0.0 <= wn <= 1.0 and 0.0 <= wg <= 1.0):
            raise ValueError("wl, wn, and wg must each be in [0, 1]")
        if not (0.0 <= inertia <= 1.0):
            raise ValueError("inertia must be in [0, 1]")

        self.inertia = float(inertia)
        self.wl = float(wl); self.wn = float(wn); self.wg = float(wg)

    @property
    def position(self) -> FloatArray:
        """Expose the base-class candidate buffer as the current position vector."""
        return self.candidate

    @position.setter
    def position(self, value: FloatArray) -> None:
        """
        Set position after shape validation and hard clipping to internal bounds.

        Why clip here
        - Ensures all externally assigned positions respect the frozen box,
          keeping the particle valid even if called by user code.
        """
        arr = np.asarray(value, dtype=float, copy=True)
        if arr.shape != (self.size,):
            raise ValueError(f"position must have shape ({self.size},); got {arr.shape}")
        np.clip(arr, self._lower, self._upper, out=arr)
        self.candidate[:] = arr

    @classmethod
    def generate(
        cls,
        *,
        size: int,
        lower: FloatArray,
        upper: FloatArray,
        inertia: float,
        wl: float,
        wn: float,
        wg: float,
        vmax_ratio: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> "ParticleCandidate":
        """
        Factory for a valid particle.

        - Samples a uniform in-bounds position.
        - Samples a symmetric velocity in [-vmax, +vmax] per dimension, where
          vmax = vmax_ratio * (upper - lower).
        """
        l_arr, u_arr, span = _normalise_bounds(lower, upper, size)
        rng = rng or np.random.default_rng()

        # Position uniformly in the hyper-rectangle.
        position = rng.uniform(l_arr, u_arr)

        # Velocity scaled to the box size to avoid explosive early steps.
        ratio = cls.VMAX_RATIO_DEFAULT if vmax_ratio is None else float(vmax_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("`vmax_ratio` must be in (0, 1]")
        vmax = ratio * span
        velocity = rng.uniform(-vmax, vmax)

        return cls(size, l_arr, u_arr, position, velocity, inertia, wl, wn, wg)

    def _clip_velocity(self, vmax: float | FloatArray) -> None:
        """
        Cap each velocity component to [-vmax, +vmax].

        - Accepts either a scalar vmax or per-dimension array.
        - Subtracts a tiny epsilon (TINY) from the high bound to reduce
          oscillation at the boundary due to exact tie conditions.
        """
        vmax_arr = np.asarray(vmax, dtype=float)
        if vmax_arr.shape not in [(), (self.size,)]:
            raise ValueError("`vmax` must be scalar or shape (size,)")
        hi = np.maximum(vmax_arr - TINY, 0.0)
        np.clip(self.velocity, -hi, hi, out=self.velocity)

    def recombine(self, local_best: "ParticleCandidate",
                  neighborhood_best: "ParticleCandidate",
                  global_best: "ParticleCandidate",
                  rng: np.random.Generator) -> None:
        """
        Backward-compatible recombination using particle objects.

        Delegates to the array-based method to avoid aliasing risks with
        mutable object references.
        """
        self.recombine_from_pos(local_best.candidate, neighborhood_best.candidate,
                                global_best.candidate, rng)

    def recombine_from_pos(self,
                           local_pos: NDArray[np.floating],
                           neigh_pos: NDArray[np.floating],
                           global_pos: NDArray[np.floating],
                           rng: np.random.Generator) -> None:
        """
        Standard PSO velocity update:

            v ← w * v + r_l * wl * (pbest_i - x)
                         + r_n * wn * (nbest_i - x)
                         + r_g * wg * (gbest - x)

        where r_* are i.i.d. U(0, 1) per component.
        """
        rl = rng.random(self.candidate.shape)
        rn = rng.random(self.candidate.shape)
        rg = rng.random(self.candidate.shape)

        cog   = rl * self.wl * (local_pos  - self.candidate)  # cognitive pull
        soc_n = rn * self.wn * (neigh_pos  - self.candidate)  # neighborhood pull
        soc_g = rg * self.wg * (global_pos - self.candidate)  # global pull

        self.velocity = self.inertia * self.velocity + cog + soc_n + soc_g

    def mutate(self, noise_sigma: float, rng: np.random.Generator | None = None) -> None:
        """
        Apply position update with reflective boundary handling and optional Gaussian noise.

        - Uses modulo arithmetic over a doubled span to account for an arbitrary
          number of bounces in closed form (no while-loops).
        - Velocity components that undergo an odd number of boundary crossings
          have their sign flipped.
        """
        rng = rng or np.random.default_rng()

        # Proposed unconstrained move.
        x_new = self.candidate + self.velocity

        # Map into [0, 2*span) per dimension; values beyond `span` are mirrored.
        span = self._upper - self._lower
        y = (x_new - self._lower) % (2.0 * span)
        mask = y > span  # True where we are in the mirrored half.

        # Reflect across the far boundary and shift back to absolute coordinates.
        x_reflect = np.where(mask, 2.0 * span - y, y) + self._lower

        # Count boundary crossings per dimension (parity determines sign flip).
        # floor_divide is safe since span > 0; works for negative numerators too.
        n = np.floor_divide(x_new - self._lower, span).astype(np.int64)

        # Flip velocity where the number of traversed spans is odd (XOR with mirror half).
        flip = np.bitwise_xor((n & 1).astype(bool), mask)
        self.velocity[flip] *= -1.0

        # Commit the reflected position (already in-bounds), then defensively clip.
        x_new = x_reflect
        self.candidate[:] = x_new
        np.clip(self.candidate, self._lower, self._upper, out=self.candidate)

        # Optional isotropic perturbation to escape local traps; re-clip.
        if noise_sigma > 0.0:
            self.candidate[:] += rng.normal(0.0, noise_sigma, size=self.candidate.shape)
            np.clip(self.candidate, self._lower, self._upper, out=self.candidate)

    def copy(self) -> "ParticleCandidate":
        """Deep copy particle state (bounds, position buffer, velocity, and PSO weights)."""
        return ParticleCandidate(
            self.size,
            np.array(self._lower, copy=True),
            np.array(self._upper, copy=True),
            self.candidate.copy(),
            self.velocity.copy(),
            self.inertia, self.wl, self.wn, self.wg,
        )

    def __repr__(self) -> str:
        """Compact preview showing dimensionality and leading coordinates of the position."""
        head = np.array2string(
        self.candidate[:4],
        precision=6,
        separator=", ",
        suppress_small=True,
        max_line_width=None,
        )

        ell = "…" if self.size > 4 else ""
        return f"<Particle dim={self.size} pos={head}{ell}>"
    