"""The gates, run against estimators whose behaviour is known analytically.

test_negative.py checks that each gate fires on hand-built input. That is not
enough on its own: a gate could fire correctly on a fixture and still be wrong
about real sampling behaviour. So these studies use textbook estimators, where
theory says exactly what should happen, and check that simcheck agrees.

Both directions are exercised. The sample mean is unbiased and its t interval
covers at the nominal rate, so every gate must stay silent. The uncorrected
variance estimator is biased by a known factor of ``(n-1)/n``, and the z interval
that ignores estimation of the scale under-covers at small n, so the gates must
fire -- and fire for the right reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from simcheck import (
    MonteCarloResult,
    assert_coverage,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
)

REPS = 2000
N = 25
TRUTH = 3.0
SIGMA = 2.0


def _study_sample_mean(reps: int = REPS, n: int = N, seed: int = 0):
    """Monte Carlo study of the sample mean with a normal-approximation interval.

    At n=25 the t critical value differs enough from 1.96 to matter, so the
    interval uses the t quantile rather than the normal one. It is hard-coded
    from tables: simcheck has no scipy dependency and should not gain one so a
    test can call ``t.ppf``.

    Args:
        reps: Replicates.
        n: Observations per replicate.
        seed: Generator seed.

    Returns:
        MonteCarloResult: The completed study.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(TRUTH, SIGMA, size=(reps, n))
    means = draws.mean(axis=1)
    errors = draws.std(axis=1, ddof=1) / np.sqrt(n)
    # t_{0.975, 24} = 2.0639, from tables. Hard-coded so the test does not
    # depend on a special-function implementation.
    critical = 2.063899
    return MonteCarloResult(
        estimates=means,
        standard_errors=errors,
        covered=np.abs(means - TRUTH) <= critical * errors,
        rejected=np.abs(means) > critical * errors,
        truth=TRUTH,
    )


def test_the_sample_mean_passes_every_gate():
    """An unbiased estimator with an honest interval must trip nothing."""
    result = _study_sample_mean()
    assert_unbiased(result, "sample mean")
    assert_coverage(result, 0.95, "sample mean t interval")
    assert_se_calibrated(result, "sample mean", tolerance=0.05)


def test_the_uncorrected_variance_is_caught_as_biased():
    """Dividing by n rather than n-1 biases the variance by a known factor.

    At n=25 the bias is -sigma^2/25 = -0.16, which a 2000-replicate study
    resolves easily. If ``assert_unbiased`` did not fire here it would be
    incapable of detecting a textbook bias.
    """
    rng = np.random.default_rng(1)
    draws = rng.normal(TRUTH, SIGMA, size=(REPS, N))
    biased = draws.var(axis=1, ddof=0)

    result = MonteCarloResult(
        estimates=biased,
        standard_errors=np.full(REPS, np.std(biased, ddof=1)),
        covered=np.ones(REPS, dtype=bool),
        rejected=np.zeros(REPS, dtype=bool),
        truth=SIGMA**2,
    )
    with pytest.raises(AssertionError, match="Monte Carlo"):
        assert_unbiased(result, "uncorrected variance")

    # And the bias is the one theory predicts: -sigma^2 / n.
    assert result.bias == pytest.approx(-(SIGMA**2) / N, abs=0.05)


def test_the_corrected_variance_is_not_flagged():
    """The same study with ddof=1 must pass, or the gate is just noisy."""
    rng = np.random.default_rng(1)
    draws = rng.normal(TRUTH, SIGMA, size=(REPS, N))
    unbiased = draws.var(axis=1, ddof=1)

    result = MonteCarloResult(
        estimates=unbiased,
        standard_errors=np.full(REPS, np.std(unbiased, ddof=1)),
        covered=np.ones(REPS, dtype=bool),
        rejected=np.zeros(REPS, dtype=bool),
        truth=SIGMA**2,
    )
    assert_unbiased(result, "corrected variance")


def test_a_normal_interval_under_covers_at_small_n():
    """Using 1.96 instead of the t quantile at n=5 must be caught.

    This is the mistake simcheck exists to catch in real code: an interval that
    looks right, is off by a correction that only matters at small samples, and
    passes any eyeball test.
    """
    rng = np.random.default_rng(2)
    n = 5
    draws = rng.normal(TRUTH, SIGMA, size=(REPS, n))
    means = draws.mean(axis=1)
    errors = draws.std(axis=1, ddof=1) / np.sqrt(n)

    result = MonteCarloResult(
        estimates=means,
        standard_errors=errors,
        covered=np.abs(means - TRUTH) <= 1.96 * errors,
        rejected=np.zeros(REPS, dtype=bool),
        truth=TRUTH,
    )
    assert_unbiased(result, "sample mean at n=5")
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_coverage(result, 0.95, "normal interval at n=5")

    # Coverage should land near 0.875, the known figure for a 1.96 interval on
    # 4 degrees of freedom. Asserting the value, not merely that it failed.
    assert result.coverage == pytest.approx(0.875, abs=0.03)


def test_the_gate_false_positive_rate_is_about_what_three_sigma_implies():
    """Running the gate on correct estimators must almost never fail.

    A gate that fires on 5% of correct code gets disabled within a week. Three
    sigma implies roughly one failure in 370; over 60 independent studies that
    is a failure probability of about 15%, so the assertion here is that at most
    one of 60 trips -- which itself holds with probability ~99%.
    """
    tripped = 0
    for seed in range(60):
        result = _study_sample_mean(reps=400, seed=seed + 100)
        try:
            assert_unbiased(result, f"seed {seed}")
        except AssertionError:
            tripped += 1
    assert tripped <= 1, f"{tripped} of 60 correct studies were flagged as biased"


def test_the_band_matches_the_textbook_binomial_interval():
    """The band is nominal +- sigmas * sqrt(p(1-p)/n), not something invented."""
    nominal, reps, sigmas = 0.95, 400, 3.0
    expected = sigmas * np.sqrt(nominal * (1 - nominal) / reps)
    low, high = binomial_band(nominal, reps, sigmas)
    assert low == pytest.approx(nominal - expected)
    assert high == pytest.approx(nominal + expected)
