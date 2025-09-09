from __future__ import annotations

import logging
import numpy as np
import matplotlib.pyplot as plt

from swarm_optimizer import ParticleSwarmOptimizer, ParticleCandidate, TINY

logging.basicConfig(level=logging.INFO)   # Enable concise per-iteration logs.


def neg_rosenbrock(p: ParticleCandidate) -> float:
    """
    Negative Rosenbrock function (so PSO can maximize it).
    The global minimum is at x = (1, 1, ..., 1) with value 0.
    """
    x = p.position
    return -sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def main() -> None:
    dim = 2

    # A moderate box prevents extreme penalties and boundary sticking.
    lb = np.full(dim, -5.0)
    ub = np.full(dim,  5.0)

    pso = ParticleSwarmOptimizer(
        fitness_func=neg_rosenbrock,
        pop_size=80,
        n_neighbors=5,
        size=dim,
        lower=lb,
        upper=ub,
        # Conservative coefficients that work well on Rosenbrock's curved valley.
        wl=0.5, wn=0.0, wg=0.5,
        w_max=0.90, w_min=0.40,
        vmax_ratio=0.20,
        noise_sigma=0.05,
        noise_decay=0.98,
        rng=np.random.default_rng(42),   # Make the demo reproducible.
        verbose=True,
    )

    best = pso.fit(n_iters=150)

    # Consistency check: the best object's objective must match the tracked best.
    best_fit_from_obj = neg_rosenbrock(best)
    print("Best position:", best.position)
    print("Sanity check (should match):",
          best_fit_from_obj, pso.global_fitness_best)

    # Report the achieved objective value at the found minimum.
    print("Rosenbrock minimum found (objective):", -pso.global_fitness_best)

    # Prepare series for a log-scaled convergence plot (numerically safe with TINY).
    its  = np.arange(len(pso.history_best))
    eps  = TINY

    gbest = np.maximum(-pso.history_best, eps)
    mean  = np.maximum(-pso.history_mean, eps)
    low   = np.maximum(mean - pso.history_std, eps)
    high  = np.maximum(mean + pso.history_std, eps)

    # Convergence diagnostics.
    plt.figure(figsize=(8, 5))
    plt.plot(its, gbest, label="Global best (objective)")
    plt.plot(its, mean,  label="Mean fitness")
    plt.fill_between(its, low, high, alpha=0.3, label="±1 σ")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
