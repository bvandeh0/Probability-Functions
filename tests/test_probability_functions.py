import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from probability_functions import (
    density,
    make_distribution,
    plot_distribution,
    probability,
    summary_statistics,
)


def test_binomial_known_probability():
    spec = make_distribution("binomial", n=10, p=0.5)
    assert probability(spec, 5, "=") == pytest.approx(0.24609375)
    assert probability(spec, 5, "<=") == pytest.approx(0.623046875)
    assert probability(spec, 5, ">=") == pytest.approx(0.623046875)


def test_poisson_complement_and_noninteger_threshold():
    spec = make_distribution("poisson", lam=3)
    assert probability(spec, 2, ">=") == pytest.approx(1 - probability(spec, 1, "<="))
    assert probability(spec, 2.5, ">=") == pytest.approx(probability(spec, 3, ">="))
    assert probability(spec, 2.5, "=") == 0


def test_normal_point_probability_and_density():
    spec = make_distribution("normal", mu=0, sigma=1)
    assert probability(spec, 0, "=") == 0.0
    assert density(spec, 0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert probability(spec, 0, "<=") == pytest.approx(0.5)
    assert probability(spec, 0, ">=") == pytest.approx(0.5)


def test_exponential_rate():
    spec = make_distribution("exponential", lam=2)
    assert summary_statistics(spec)["mean"] == pytest.approx(0.5)
    assert probability(spec, 1, "<=") == pytest.approx(1 - np.exp(-2))
    assert density(spec, 0) == pytest.approx(2)


def test_geometric_counts_trials_to_first_success():
    spec = make_distribution("geometric", p=0.2)
    assert probability(spec, 1, "=") == pytest.approx(0.2)
    assert probability(spec, 2, "=") == pytest.approx(0.16)


def test_hypergeometric_correct_lower_bound():
    # Only two failures are in the population, so three draws need one success.
    spec = make_distribution("hypergeometric", N=5, K=3, n=3)
    assert probability(spec, 0, "=") == 0.0
    assert probability(spec, 1, "=") == pytest.approx(0.3)
    assert probability(spec, 3, "=") == pytest.approx(0.1)


def test_negative_binomial_counts_failures():
    spec = make_distribution("negative_binomial", r=2, p=0.5)
    assert probability(spec, 0, "=") == pytest.approx(0.25)
    assert probability(spec, 1, "=") == pytest.approx(0.25)


def test_uniform_is_parameterized_by_endpoints():
    spec = make_distribution("uniform", a=2, b=6)
    assert probability(spec, 4, "<=") == pytest.approx(0.5)
    assert density(spec, 4) == pytest.approx(0.25)
    assert summary_statistics(spec)["mean"] == pytest.approx(4)


@pytest.mark.parametrize("name,params", [
    ("binomial", {"n": -1, "p": 0.5}),
    ("binomial", {"n": 3, "p": 2}),
    ("normal", {"mu": 0, "sigma": 0}),
    ("exponential", {"lam": -0.5}),
    ("hypergeometric", {"N": 5, "K": 6, "n": 2}),
    ("uniform", {"a": 5, "b": 2}),
])
def test_invalid_parameters_raise(name, params):
    with pytest.raises(ValueError):
        make_distribution(name, **params)


def test_all_distributions_can_plot_without_opening_window():
    cases = [
        ("binomial", {"n": 10, "p": 0.5}),
        ("poisson", {"lam": 3}),
        ("normal", {"mu": 0, "sigma": 1}),
        ("exponential", {"lam": 2}),
        ("geometric", {"p": 0.3}),
        ("hypergeometric", {"N": 20, "K": 5, "n": 6}),
        ("negative_binomial", {"r": 2, "p": 0.5}),
        ("uniform", {"a": 2, "b": 6}),
    ]
    import matplotlib.pyplot as plt
    for name, params in cases:
        fig, ax = plot_distribution(make_distribution(name, **params), show=False)
        assert ax.get_title()
        assert len(ax.lines) + len(ax.patches) > 0
        plt.close(fig)


def test_reject_invalid_comparison():
    with pytest.raises(ValueError):
        probability(make_distribution("binomial", n=2, p=0.5), 1, ">")


def test_noninteger_discrete_greater_equal():
    spec = make_distribution("binomial", n=4, p=0.5)
    assert probability(spec, 1.5, ">=") == pytest.approx(probability(spec, 2, ">="))
