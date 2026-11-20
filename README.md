# Probability Functions

An interactive Python probability calculator and distribution visualizer using **SciPy**, **NumPy**, and **Matplotlib**.

Supports eight distributions: Binomial, Poisson, Normal, Exponential, Geometric, Hypergeometric, Negative Binomial, and Uniform.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/bvandeh0/Probability-Functions.git
cd Probability-Functions
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python probability_functions.py
```

Select a distribution, enter its parameters, and choose `=`, `<=`, or `>=` to calculate a probability. The optional plot uses Matplotlib.

## Interactive Jupyter notebook

See **[Probability distributions: calculations, visualizations, and applications](notebooks/probability_distributions.ipynb)** for a hands-on walkthrough of all eight distributions, with theoretical formulas, numerical examples, plots, interpretation, and a reproducible simulation.

Launch from the repository root:

```bash
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Then open `notebooks/probability_distributions.ipynb` and use **Run All**. Every example is hypothetical; the notebook does not use proprietary or personal data.

## Python API

```python
from probability_functions import make_distribution, probability, density, summary_statistics

spec = make_distribution("binomial", n=10, p=0.5)
print(probability(spec, 5, "="))  # 0.24609375
print(probability(spec, 5, ">="))
print(summary_statistics(spec))

uniform = make_distribution("uniform", a=2, b=6)
print(density(uniform, 4))         # 0.25
```

To create a plot:

```python
from probability_functions import make_distribution, plot_distribution

plot_distribution(make_distribution("normal", mu=0, sigma=1))
```

The original `plot_binomial`, `plot_normal`, `plot_uniform`, and other distribution plotting wrappers are retained.

## Parameter definitions

| Distribution | Inputs | Random variable |
| --- | --- | --- |
| Binomial | `n` (number of trials), `p` (success probability) | Number of successes |
| Poisson | `lam` (rate/mean count) | Event count |
| Normal | `mu` (mean), `sigma` (standard deviation) | Real-valued observation |
| Exponential | `lam` (rate, **not scale**) | Waiting time |
| Geometric | `p` (success probability) | Number of trials up to first success |
| Hypergeometric | `N` (population size), `K` (population successes), `n` (draws) | Number of successes drawn without replacement |
| Negative Binomial | `r` (required successes), `p` (success probability) | **Failures before** the `r`th success |
| Uniform | `a` (lower endpoint), `b` (upper endpoint) | Value between `a` and `b` |

For discrete distributions, `=` gives the probability mass. For continuous distributions, an exact point has probability **zero**; use `density(spec, x)` for its PDF value. The comparison `>=` is inclusive (including the boundary mass for discrete distributions).

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests include known numerical values, invalid parameter validation, and headless graph creation for all eight distributions.

## Changes from initial version

This revision removes duplicate function definitions, fixes syntax and an unfinished exponential plot, corrects the uniform distribution parameterization, clarifies the negative binomial interpretation, and uses an iterative CLI rather than recursive menu calls.
