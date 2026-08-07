"""The runner's guarantees, each with the failure it prevents.

Three properties matter here and none of them is obvious from reading the loop:
replicates must be independently reproducible, a study that recorded no
intervals must not be confused with one whose intervals never cover, and an
estimator that reports intervals only sometimes must be rejected rather than
averaged.
"""

from __future__ import annotations

import numpy as np
import pytest

from simcheck import (
    Estimate,
    assert_coverage,
    assert_unbiased,
    monte_carlo,
)

TRUTH = 2.0
N = 40


def _mean_with_interval(rng: np.random.Generator) -> Estimate:
    """Sample mean of 40 normal draws with a two-standard-error interval."""
    x = rng.normal(TRUTH, 1.0, N)
    se = float(x.std(ddof=1) / np.sqrt(N))
    return Estimate(float(x.mean()), se, x.mean() - 1.96 * se, x.mean() + 1.96 * se)


def _mean_without_interval(rng: np.random.Generator) -> Estimate:
    """The same estimator, reporting no interval."""
    x = rng.normal(TRUTH, 1.0, N)
    return Estimate(float(x.mean()), float(x.std(ddof=1) / np.sqrt(N)))


# --------------------------------------------------------------------------
# Reproducibility.
# --------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_study():
    """Otherwise a failure cannot be investigated."""
    a = monte_carlo(_mean_with_interval, TRUTH, 200, seed=7)
    b = monte_carlo(_mean_with_interval, TRUTH, 200, seed=7)
    np.testing.assert_array_equal(a.estimates, b.estimates)


def test_different_seeds_give_different_studies():
    """The guard on the guard: identical output would mean the seed is ignored."""
    a = monte_carlo(_mean_with_interval, TRUTH, 200, seed=7)
    b = monte_carlo(_mean_with_interval, TRUTH, 200, seed=8)
    assert not np.array_equal(a.estimates, b.estimates)


def test_raising_the_replicate_count_extends_the_study_rather_than_changing_it():
    """Replicate ``i`` must depend on ``(seed, i)`` alone.

    Drawing every replicate from one shared stream makes replicate 400 depend on
    every draw before it. Then a 400-replicate study and a 4000-replicate study
    share nothing, a single replicate cannot be reproduced on its own, and a
    result that moved when the count changed is impossible to attribute. Seed
    spawning is what buys this, and it is invisible unless asserted.
    """
    short = monte_carlo(_mean_with_interval, TRUTH, 50, seed=3)
    long = monte_carlo(_mean_with_interval, TRUTH, 500, seed=3)
    np.testing.assert_array_equal(short.estimates, long.estimates[:50])


# --------------------------------------------------------------------------
# A missing interval is not a missed one.
# --------------------------------------------------------------------------


def test_a_study_without_intervals_has_no_coverage_rather_than_zero_coverage():
    """Zero coverage and no coverage are opposite conclusions.

    Recording ``covered = False`` when the estimator reported no interval makes
    coverage come out at 0.000, and the assertion then fails with "outside the
    band" -- which reads as a catastrophically broken estimator rather than as a
    study that never measured coverage at all.
    """
    result = monte_carlo(_mean_without_interval, TRUTH, 100, seed=0)

    assert result.covered is None
    with pytest.raises(ValueError, match="recorded no confidence intervals"):
        _ = result.coverage
    with pytest.raises(ValueError, match="recorded no confidence intervals"):
        assert_coverage(result, 0.95, "no intervals")

    # The rest of the study is still usable.
    assert_unbiased(result, "sample mean without intervals")


def test_a_study_without_decisions_has_no_rejection_rate():
    """Same argument for size and power."""
    result = monte_carlo(_mean_without_interval, TRUTH, 50, seed=0)
    assert result.rejected is None
    with pytest.raises(ValueError, match="no reject/accept decisions"):
        _ = result.rejection_rate


def test_reporting_an_interval_only_sometimes_is_rejected():
    """Averaging over a mixture of two estimators is not a coverage rate."""

    def sometimes(rng: np.random.Generator) -> Estimate:
        x = rng.normal(TRUTH, 1.0, N)
        se = float(x.std(ddof=1) / np.sqrt(N))
        if x.mean() > TRUTH:
            return Estimate(float(x.mean()), se, x.mean() - se, x.mean() + se)
        return Estimate(float(x.mean()), se)

    with pytest.raises(ValueError, match="reported an interval and the rest did not"):
        monte_carlo(sometimes, TRUTH, 100, seed=0)


def test_reporting_a_decision_only_sometimes_is_rejected():
    """And the same for the rejection flag."""

    def sometimes(rng: np.random.Generator) -> Estimate:
        x = rng.normal(TRUTH, 1.0, N)
        return Estimate(float(x.mean()), rejected=True if x.mean() > TRUTH else None)

    with pytest.raises(ValueError, match="reported a reject/accept decision"):
        monte_carlo(sometimes, TRUTH, 100, seed=0)


# --------------------------------------------------------------------------
# The runner reproduces what the gates expect.
# --------------------------------------------------------------------------


def test_the_runner_recovers_the_nominal_coverage_of_a_correct_interval():
    """End to end: a 1.96 interval at n=40 covers at about 95%."""
    result = monte_carlo(_mean_with_interval, TRUTH, 2000, seed=11)
    assert_unbiased(result, "sample mean")
    assert_coverage(result, 0.95, "1.96 interval at n=40")


def test_the_runner_catches_an_interval_that_is_too_narrow():
    """Halving the interval must show up as under-coverage."""

    def narrow(rng: np.random.Generator) -> Estimate:
        x = rng.normal(TRUTH, 1.0, N)
        se = float(x.std(ddof=1) / np.sqrt(N))
        return Estimate(float(x.mean()), se, x.mean() - se, x.mean() + se)

    result = monte_carlo(narrow, TRUTH, 2000, seed=11)
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_coverage(result, 0.95, "one-standard-error interval")
    # A +-1 SE interval covers about 68% of the time.
    assert result.coverage == pytest.approx(0.68, abs=0.03)


def test_zero_replicates_is_rejected():
    """An empty study would make every rate a division by zero."""
    with pytest.raises(ValueError, match="reps must be positive"):
        monte_carlo(_mean_with_interval, TRUTH, 0)
