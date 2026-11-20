"""Interactive probability calculator and visualization utilities.

Distributions use SciPy parameter conventions unless documented otherwise.
For the negative binomial, k counts failures before r successes.
For continuous distributions, equality returns probability zero (the density is
reported separately); for discrete distributions equality returns the PMF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    distribution: Any
    discrete: bool
    description: str


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return value


def _probability(value: float, label: str = "p") -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{label} must be between 0 (exclusive) and 1 (inclusive).")
    return value


def _integer(value: float, label: str, minimum: int = 0) -> int:
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(number)


def make_distribution(name: str, **parameters: float) -> DistributionSpec:
    """Construct a validated frozen SciPy distribution.

    Parameters:
        binomial: n (trials), p (success probability)
        poisson: lam (mean count)
        normal: mu (mean), sigma (standard deviation)
        exponential: lam (rate), not scale
        geometric: p (success probability), X counts trials until first success
        hypergeometric: N (population), K (successes in population), n (draws)
        negative_binomial: r (successes), p (success probability),
            X counts failures before r successes
        uniform: a (lower endpoint), b (upper endpoint)
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key == "binomial":
        n = _integer(parameters["n"], "n")
        p = _probability(parameters["p"])
        return DistributionSpec("Binomial", stats.binom(n=n, p=p), True, f"n={n}, p={p}")
    if key == "poisson":
        lam = _positive(parameters["lam"], "lam")
        return DistributionSpec("Poisson", stats.poisson(mu=lam), True, f"lam={lam}")
    if key == "normal":
        mu = float(parameters["mu"])
        sigma = _positive(parameters["sigma"], "sigma")
        if not np.isfinite(mu):
            raise ValueError("mu must be finite.")
        return DistributionSpec("Normal", stats.norm(loc=mu, scale=sigma), False,
                                f"mu={mu}, sigma={sigma}")
    if key == "exponential":
        lam = _positive(parameters["lam"], "lam")
        return DistributionSpec("Exponential", stats.expon(scale=1 / lam), False,
                                f"rate={lam}")
    if key == "geometric":
        p = _probability(parameters["p"])
        return DistributionSpec("Geometric", stats.geom(p=p), True, f"p={p}")
    if key == "hypergeometric":
        N = _integer(parameters["N"], "N", minimum=1)
        K = _integer(parameters["K"], "K")
        n = _integer(parameters["n"], "n")
        if K > N or n > N:
            raise ValueError("K and n cannot exceed population size N.")
        return DistributionSpec("Hypergeometric", stats.hypergeom(M=N, n=K, N=n), True,
                                f"N={N}, K={K}, n={n}")
    if key in ("negative_binomial", "negativebinomial"):
        r = _integer(parameters["r"], "r", minimum=1)
        p = _probability(parameters["p"])
        return DistributionSpec("Negative Binomial", stats.nbinom(n=r, p=p), True,
                                f"r={r}, p={p}")
    if key == "uniform":
        a, b = float(parameters["a"]), float(parameters["b"])
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("Uniform requires finite endpoints with b > a.")
        return DistributionSpec("Uniform", stats.uniform(loc=a, scale=b - a),
                                False, f"a={a}, b={b}")
    raise ValueError(f"Unknown distribution: {name}")


def probability(spec: DistributionSpec, x: float, comparison: str = "=") -> float:
    """Return P(X = x), P(X <= x), or P(X >= x).

    Equality is 0 for continuous distributions; use density() for their PDF.
    For integer-valued distributions, the strict complement accurately handles
    non-integer thresholds as well as integer ones.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("x must be finite.")
    if comparison == "=":
        if not spec.discrete:
            return 0.0
        return float(spec.distribution.pmf(x))
    if comparison == "<=":
        return float(spec.distribution.cdf(x))
    if comparison == ">=":
        if spec.discrete:
            return float(spec.distribution.sf(np.ceil(x) - 1))
        return float(spec.distribution.sf(x))
    raise ValueError("comparison must be '=', '<=', or '>='.")


def density(spec: DistributionSpec, x: float) -> float:
    """PMF at x (discrete) or PDF at x (continuous)."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("x must be finite.")
    return float(spec.distribution.pmf(x) if spec.discrete
                 else spec.distribution.pdf(x))


def summary_statistics(spec: DistributionSpec) -> dict[str, float]:
    """Return distribution mean, variance and skewness."""
    mean, variance, skew = spec.distribution.stats(moments="mvs")
    return {"mean": float(mean), "variance": float(variance), "skewness": float(skew)}


def plot_distribution(spec: DistributionSpec, *, maximum: int | None = None,
                      show: bool = True):
    """Plot a sensible support window; return the Matplotlib figure and axis."""
    dist = spec.distribution
    lower, upper = dist.support()
    if spec.discrete:
        # Quantile clipping avoids enormous bars for unbounded distributions.
        left = max(lower, dist.ppf(0.001)) if np.isfinite(lower) else dist.ppf(0.001)
        right = min(upper, dist.ppf(0.999)) if np.isfinite(upper) else dist.ppf(0.999)
        left = int(max(0, np.floor(left)))
        right = int(np.ceil(right))
        if maximum is not None:
            right = _integer(maximum, "maximum")
        # Bound the visualization, not the underlying probability computation.
        right = min(right, left + 250)
        if right < left:
            raise ValueError("maximum must not be below the distribution's lower support.")
        x = np.arange(left, right + 1)
        y = dist.pmf(x)
    else:
        left = float(lower) if np.isfinite(lower) else float(dist.ppf(0.001))
        right = float(upper) if np.isfinite(upper) else float(dist.ppf(0.999))
        x = np.linspace(left, right, 500)
        y = dist.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    if spec.discrete:
        ax.bar(x, y, width=0.85)
        ax.set_ylabel("Probability mass")
    else:
        ax.plot(x, y)
        ax.set_ylabel("Probability density")
    ax.set_title(f"{spec.name} distribution ({spec.description})")
    ax.set_xlabel("x")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


# Backward-compatible plotting functions from the original repository.
def plot_binomial(n, p):
    return plot_distribution(make_distribution("binomial", n=n, p=p))


def plot_poisson(lam, k=None):
    return plot_distribution(make_distribution("poisson", lam=lam), maximum=k)


def plot_normal(mu, sigma):
    return plot_distribution(make_distribution("normal", mu=mu, sigma=sigma))


def plot_exponential(lam):
    return plot_distribution(make_distribution("exponential", lam=lam))


def plot_geometric(p):
    return plot_distribution(make_distribution("geometric", p=p))


def plot_hypergeometric(N, K, n):
    return plot_distribution(make_distribution("hypergeometric", N=N, K=K, n=n))


def plot_negative_binomial(p, r):
    return plot_distribution(make_distribution("negative_binomial", p=p, r=r))


def plot_uniform(a, b):
    return plot_distribution(make_distribution("uniform", a=a, b=b))


_OPTIONS = {
    "1": ("binomial", ("n", "p")),
    "2": ("poisson", ("lam",)),
    "3": ("normal", ("mu", "sigma")),
    "4": ("exponential", ("lam",)),
    "5": ("geometric", ("p",)),
    "6": ("hypergeometric", ("N", "K", "n")),
    "7": ("negative_binomial", ("r", "p")),
    "8": ("uniform", ("a", "b")),
}


def _interactive_calculation(name: str, parameter_names: tuple[str, ...]) -> None:
    """Run one calculator session without recursive menu calls."""
    try:
        values = {}
        for param in parameter_names:
            raw = input(f"Enter {param} (blank to return): ").strip()
            if not raw:
                return
            values[param] = float(raw)
        spec = make_distribution(name, **values)
        raw_x = input("Enter x (blank to return): ").strip()
        if not raw_x:
            return
        x = float(raw_x)
        comparison = input("Choose comparison (=, <=, >=): ").strip()
        result = probability(spec, x, comparison)
        print(f"P(X {comparison} {x:g}) = {result:.8g}")
        if not spec.discrete:
            print(f"PDF at x={x:g}: {density(spec, x):.8g}")
        statistics = summary_statistics(spec)
        print("Mean: {mean:.6g}; Variance: {variance:.6g}; "
              "Skewness: {skewness:.6g}".format(**statistics))
        if input("Plot distribution? [y/N]: ").strip().lower() == "y":
            plot_distribution(spec)
    except (ValueError, OverflowError) as exc:
        print(f"Invalid input: {exc}")


# Compatibility names for the original interactive entry points.
def binomial():
    _interactive_calculation(*_OPTIONS["1"])


def pois():
    _interactive_calculation(*_OPTIONS["2"])


def normal():
    _interactive_calculation(*_OPTIONS["3"])


def exponential():
    _interactive_calculation(*_OPTIONS["4"])


def geometric():
    _interactive_calculation(*_OPTIONS["5"])


def hypergeometric():
    _interactive_calculation(*_OPTIONS["6"])


def negative_binomial():
    _interactive_calculation(*_OPTIONS["7"])


def unif():
    _interactive_calculation(*_OPTIONS["8"])


def main():
    """Terminal menu for probability calculations and distribution plots."""
    while True:
        print("\nProbability Calculator")
        for number, (name, _) in _OPTIONS.items():
            print(f"  {number}. {name.replace('_', ' ').title()}")
        print("  exit. Quit")
        choice = input("Choose a distribution: ").strip().lower()
        if choice in ("exit", "quit", "q"):
            print("Goodbye!")
            return
        if choice in _OPTIONS:
            _interactive_calculation(*_OPTIONS[choice])
        else:
            print("Invalid choice; choose 1-8 or exit.")


if __name__ == "__main__":
    main()
