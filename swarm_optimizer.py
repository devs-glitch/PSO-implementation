from __future__ import annotations
import logging
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from pso_candidate import ParticleCandidate, FloatArray, _normalise_bounds, TINY

FitnessScalar = Callable[[ParticleCandidate], float]
FitnessVector = Callable[[NDArray[np.floating]], NDArray[np.floating]]


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (maximization form).

    Inertia schedule
    - Inertia begins at `w_max` and decays quadratically to `w_min`
      across `n_iters` iterations (inclusive of index n_iters - 1), unless
      a custom `w_schedule` is provided.
    """

    def __init__(
        self,
        fitness_func: FitnessScalar | FitnessVector,
        pop_size: int,
        n_neighbors: int,
        *,
        size: int,
        lower: FloatArray,
        upper: FloatArray,
        wl: float = 0.25,
        wn: float = 0.15,
        wg: float = 0.60,
        w_max: float = 0.95,
        w_min: float = 0.60,
        vmax_ratio: float = 0.4,
        w_schedule: Callable[[float], float] | None = None,
        noise_sigma: float = 0.05,
        noise_decay:  float = 0.995,
        sigma_cutoff: float | None = None,
        vectorized: bool | None = None,
        rng: np.random.Generator | None = None,
        verbose: bool = True,
    ) -> None:

        # Vectorization mode: allow passing a function that evaluates a batch X (pop x dim).
        self._vectorized = bool(getattr(fitness_func, "_vectorized", False)) if vectorized is None else bool(vectorized)
        self.fitness_func = fitness_func

        # Core sizes must be positive integers.
        if not isinstance(pop_size, int) or pop_size < 1:
            raise ValueError(f"`pop_size` must be a positive integer; got {pop_size!r}")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"`size` must be a positive integer; got {size!r}")
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("`n_neighbors` must be a positive integer (≥ 1)")

        self.pop_size = pop_size
        self.n_neighbors = n_neighbors
        self.size = size
        self._rng = rng or np.random.default_rng()

        # Validate and cache bounds and span (used for velocity caps and reflection).
        self._lower, self._upper, self._span = _normalise_bounds(lower, upper, size)

        # PSO weights (convex combination constraint).
        if not np.isclose(wl + wn + wg, 1.0): raise ValueError("wl + wn + wg must sum to 1")
        if not all(w >= 0.0 for w in (wl, wn, wg)): raise ValueError("wl, wn and wg must all be non-negative")
        if not (0.0 < w_min <= 1.0): raise ValueError("w_min must be in (0, 1]")
        if not (0.0 < w_max <= 1.0): raise ValueError("w_max must be in (0, 1]")
        if w_max < w_min: raise ValueError("w_max must be ≥ w_min")
        self.w_max = float(w_max); self.w_min = float(w_min)

        # Default inertia schedule: quadratic decay from w_max to w_min.
        self._w_schedule = (lambda frac: w_max - (w_max - w_min) * frac**2) if w_schedule is None else w_schedule

        # Velocity cap as a fraction of span (tapers over time in the loop).
        self._vmax_ratio = float(vmax_ratio)
        if not (0.0 < self._vmax_ratio <= 1.0):
            raise ValueError("`vmax_ratio` must be in (0, 1]")

        # Noise configuration (annealed Gaussian).
        if noise_sigma < 0.0:
            raise ValueError("`noise_sigma` must be non-negative")
        if not (0.0 <= noise_decay <= 1.0):
            raise ValueError("`noise_decay` must lie in [0, 1]")

        # Default terminal cutoff for noise (not below numerical TINY).
        if sigma_cutoff is None:
            sigma_cutoff = max(noise_sigma * 1e-3, TINY)

        # If noise is used, cutoff must be strictly smaller than the initial sigma.
        if noise_sigma > 0.0 and not (0.0 < sigma_cutoff < noise_sigma):
            raise ValueError("`sigma_cutoff` must be positive and smaller than `noise_sigma`")

        self._sigma0 = float(noise_sigma)
        self._sigma_decay = float(noise_decay)
        self._sigma_cut = float(sigma_cutoff)

        # Compute the iteration at which noise is considered zero.
        # sigma_t = sigma0 * decay**t < cutoff  →  t > log(cutoff/sigma0)/log(decay)
        if self._sigma0 == 0.0:
            self._noise_stop_iter = 0                     # no noise at all
        elif self._sigma_decay == 1.0:
            self._noise_stop_iter = np.inf                # constant noise (never below cutoff)
        elif self._sigma_decay == 0.0:
            self._noise_stop_iter = 1                     # one-shot noise at t=0 only
        else:
            t_star = np.log(self._sigma_cut / self._sigma0) / np.log(self._sigma_decay)
            # Guarding against tiny negative due to rounding (max with 0).
        self._noise_stop_iter = int(np.ceil(max(t_star, 0.0)))
        # ⚠︎ Note: As written, the final assignment references `t_star` unconditionally.
        # If any of the *earlier* branches executed, `t_star` is undefined.
        # (Behavioral note, not a change.)

        # Initialize population with consistent hyperparameters.
        self.population = [
            ParticleCandidate.generate(
                size=size, lower=lower, upper=upper,
                inertia=w_max, wl=wl, wn=wn, wg=wg,
                vmax_ratio=vmax_ratio, rng=self._rng
            ) for _ in range(pop_size)
        ]

        # Personal bests: store both arrays (fast) and objects (compatibility).
        self.best_pos = np.stack([p.position.copy() for p in self.population])  # shape: (pop, dim)
        self.best = [p.copy() for p in self.population]
        self.fitness_best = np.full(pop_size, -np.inf)

        # Global best tracked as both object and position array.
        self.global_best: Optional[ParticleCandidate] = None
        self.gbest_pos = None
        self.global_fitness_best = -np.inf

        # Lightweight training history for diagnostics / plots.
        self._history_best: list[float] = []
        self._history_mean: list[float] = []
        self._history_std: list[float] = []

        # Neighborhood graph (indices per particle).
        self._neigh_idx: list[np.ndarray] = []

        # Logging (INFO prints iteration summary if verbose=True).
        self._log = logging.getLogger(f"{__name__}.PSO")
        self._log.propagate = True
        self._log.setLevel(logging.INFO if verbose else logging.WARNING)

    def _record_stats(self, best: float, mean: float, std: float) -> None:
        """Append scalar summary statistics for this iteration."""
        self._history_best.append(best)
        self._history_mean.append(mean)
        self._history_std.append(std)

    @property
    def history_best(self) -> NDArray[np.floating]:
        """np.array view over best-so-far fitness values per iteration."""
        return np.array(self._history_best, dtype=float)

    @property
    def history_mean(self) -> NDArray[np.floating]:
        """np.array view over mean fitness values per iteration."""
        return np.array(self._history_mean, dtype=float)

    @property
    def history_std(self) -> NDArray[np.floating]:
        """np.array view over population fitness standard deviations per iteration."""
        return np.array(self._history_std, dtype=float)

    def _evaluate_population(self) -> NDArray[np.floating]:
        """
        Compute fitness for all particles, handling vectorized or scalar callables.

        - In vectorized mode, `fitness_func` must return a 1D array of length `pop_size`.
        - Non-finite values (NaN, ±Inf) are treated as worst possible (-Inf) since we maximize.
        """
        if self._vectorized:
            X = np.stack([p.position for p in self.population], dtype=float)
            raw = self.fitness_func(X)
            raw = np.asarray(raw, dtype=float).squeeze()
            if raw.ndim != 1:
                if raw.ndim == 2 and 1 in raw.shape:
                    raw = raw.reshape(-1)
                else:
                    raise ValueError(
                        f"Vectorised fitness must be 1D of length {self.pop_size}; got shape {np.asarray(raw).shape}"
                    )
            if raw.shape[0] != self.pop_size:
                raise ValueError(
                    f"Vectorised fitness length must be {self.pop_size}; got {raw.shape[0]}"
                )
        else:
            raw = np.array([self.fitness_func(p) for p in self.population], dtype=float)

        # Robustness: penalize invalid numbers uniformly.
        raw = np.nan_to_num(raw, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        return raw

    def fit(self, n_iters: int) -> ParticleCandidate:
        """
        Run the PSO loop for `n_iters` iterations and return the best particle object.

        Returns ParticleCandidate
            A copy-synchronized particle whose `position` equals the best-known
            position at termination (also accessible via `gbest_pos`).
        """
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("`n_iters` must be a positive integer (≥ 1)")

        # Precompute inertia schedule along [0, 1] (inclusive).
        frac_vec = np.array([0.0], dtype=float) if n_iters == 1 else np.linspace(0.0, 1.0, n_iters)
        w_schedule = np.array([self._w_schedule(f) for f in frac_vec], dtype=float)

        # Initial evaluation and bests.
        fitness = self._evaluate_population()
        self.fitness_best[:] = fitness
        idx0 = int(np.argmax(fitness))
        self.gbest_pos = self.best_pos[idx0].copy()
        self.global_best = self.best[idx0].copy()
        self.global_fitness_best = fitness[idx0]
        self._record_stats(self.global_fitness_best, float(np.mean(fitness)), float(np.std(fitness)))

        # Neighborhood construction: random fixed-degree (without self).
        self._neigh_idx = []
        idxs = np.arange(self.pop_size)
        for i in idxs:
            if self.pop_size == 1:
                self._neigh_idx.append(np.array([i], dtype=int)); continue
            others = np.delete(idxs, i)
            neigh = others if self.n_neighbors >= others.size else self._rng.choice(others, size=self.n_neighbors, replace=False)
            self._neigh_idx.append(neigh)

        # Main optimization loop.
        for it in range(n_iters):
            frac_now = float(frac_vec[it])            # 0 → 1 progress
            w = float(w_schedule[it])                 # inertia at this iteration
            sigma_now = 0.0 if it >= self._noise_stop_iter else self._sigma0 * self._sigma_decay**it

            # Velocity update + move for each particle.
            for i, p in enumerate(self.population):
                p.inertia = w
                neigh = self._neigh_idx[i]
                j = int(neigh[np.argmax(self.fitness_best[neigh])])  # best neighbor index

                # Use array forms of pbest/nbest/gbest to avoid object aliasing costs.
                p.recombine_from_pos(self.best_pos[i], self.best_pos[j], self.gbest_pos, rng=self._rng)

                # Taper velocity cap as iterations progress to encourage convergence.
                vmax_now = self._vmax_ratio * self._span * (1 - frac_now)
                p._clip_velocity(vmax_now)

                # Reflective move with annealed Gaussian noise.
                p.mutate(noise_sigma=sigma_now, rng=self._rng)

            # Evaluate new population and update personal bests where improved.
            fitness = self._evaluate_population()
            better = fitness > self.fitness_best
            for k in np.nonzero(better)[0]:
                self.fitness_best[k] = fitness[k]
                self.best_pos[k] = self.population[k].position.copy()
                self.best[k] = self.population[k].copy()

            # Update global best if any personal best beats it.
            idx = int(np.argmax(self.fitness_best))
            if self.fitness_best[idx] > self.global_fitness_best:
                self.global_fitness_best = self.fitness_best[idx]
                self.gbest_pos = self.best_pos[idx].copy()
                self.global_best = self.best[idx].copy()

            # Record diagnostics and log a compact iteration summary.
            self._record_stats(self.global_fitness_best, float(np.mean(fitness)), float(np.std(fitness)))
            self._log.info("Iter %3d/%d | Best: %.6e | Mean: %.6e", it, n_iters, self.global_fitness_best, self.history_mean[-1])

        # Final consistency: ensure the returned object is synchronized with gbest_pos.
        idx = int(np.argmax(self.fitness_best))
        if (self.global_best is None) or (self.fitness_best[idx] >= self.global_fitness_best):
            self.global_fitness_best = self.fitness_best[idx]
            self.gbest_pos = self.best_pos[idx].copy()
            self.global_best = self.best[idx].copy()
        self.global_best.position = self.gbest_pos.copy()
        return self.global_best


__all__ = ["ParticleSwarmOptimizer", "ParticleCandidate", "TINY"]
