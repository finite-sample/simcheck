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

import math

import numpy as np
import pytest

from simcheck import (
    Estimate,
    MonteCarloResult,
    assert_coverage,
    assert_intervals_informative,
    assert_more_powerful,
    assert_narrower,
    assert_power,
    assert_proportion,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
    monte_carlo,
    width_ratio,
)

REPS = 2000
N = 25
TRUTH = 3.0
SIGMA = 2.0
# The two-sided 95% normal quantile, and the t quantile on 4 degrees of freedom,
# from tables. simcheck has no scipy dependency and should not gain one so a test
# can call `norm.ppf`.
Z_95 = 1.959964
T_95_DF4 = 2.776445


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function.

    Args:
        x: Where to evaluate it.

    Returns:
        float: ``P(Z <= x)``.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _z_test_power(delta: float, n: int, sigma: float = 1.0) -> float:
    """Textbook power of a two-sided z test of a zero mean.

    Args:
        delta: True mean under the alternative.
        n: Observations per replicate.
        sigma: Known standard deviation.

    Returns:
        float: The probability of rejecting.
    """
    shift = delta * math.sqrt(n) / sigma
    return _normal_cdf(-Z_95 + shift) + _normal_cdf(-Z_95 - shift)


def _z_test_study(delta: float, n: int, reps: int = REPS, seed: int = 0):
    """Monte Carlo study of a two-sided z test with a known scale.

    Args:
        delta: True mean under the alternative.
        n: Observations per replicate.
        reps: Replicates.
        seed: Seed for the replicate stream.

    Returns:
        MonteCarloResult: The completed study.
    """
    error = 1.0 / math.sqrt(n)

    def replicate(rng: np.random.Generator) -> Estimate:
        mean = float(rng.normal(delta, 1.0, n).mean())
        return Estimate(
            mean,
            error,
            mean - Z_95 * error,
            mean + Z_95 * error,
            rejected=abs(mean) > Z_95 * error,
        )

    return monte_carlo(replicate, delta, reps, seed=seed)


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


def test_the_power_of_a_z_test_is_the_one_the_formula_gives():
    """Power has a closed form, so the gate can be checked against it.

    For a two-sided z test at the 5% level, ``Phi(-1.96 + delta*sqrt(n)/sigma) +
    Phi(-1.96 - delta*sqrt(n)/sigma)``. At delta=0.3 and n=25 that is 0.323. A
    gate for power that could not recover a number this well known would be
    measuring something else.
    """
    expected = _z_test_power(0.3, 25)
    assert expected == pytest.approx(0.3230, abs=1e-4)

    study = _z_test_study(0.3, 25, seed=31)
    assert_proportion(study.rejection_rate, study.reps, expected, "z test at n=25")
    assert_power(study, expected, "z test at n=25")

    # And it is genuinely underpowered against a claim of one half.
    with pytest.raises(AssertionError, match="below the one-sided"):
        assert_power(study, 0.50, "z test at n=25")


def test_a_sixteen_fold_sample_is_caught_as_more_powerful():
    """Power rises with n by a known amount, and the paired gate must see it.

    From 0.323 at n=25 to 0.851 at n=100, both from the formula. Asserting
    ``strong > weak`` would also pass here -- and would equally have passed on a
    gap of one replicate, which is the failure this replaces.
    """
    assert _z_test_power(0.3, 100) == pytest.approx(0.8508, abs=1e-4)

    strong = _z_test_study(0.3, 100, seed=32)
    weak = _z_test_study(0.3, 25, seed=33)
    assert_more_powerful(strong, weak, "n=100 against n=25")

    # Two studies of the same test differ only by noise, and must not be ranked.
    with pytest.raises(AssertionError, match="not measurably above"):
        assert_more_powerful(
            _z_test_study(0.3, 25, seed=34), _z_test_study(0.3, 25, seed=35), "same"
        )


def test_an_interval_at_the_known_scale_is_exactly_as_wide_as_it_must_be():
    """With sigma known the z interval is the oracle, so its width ratio is one.

    This pins the denominator of :func:`width_ratio`: if the reference width were
    off by a factor, this number would not come out at one.
    """
    study = _z_test_study(0.0, 25, seed=36)
    assert width_ratio(study, 0.95) == pytest.approx(1.0, abs=0.05)
    assert_coverage(study, 0.95, "z interval at the known scale")
    assert_intervals_informative(study, 0.95, "z interval at the known scale")


def test_the_normal_interval_is_narrower_than_the_t_interval_and_that_is_the_defect():
    """Narrower is not better, and the textbook case says why.

    At n=5 the 1.96 interval is ``1.96 / 2.776 = 0.706`` times the width of the t
    interval, which :func:`assert_narrower` confirms -- and it covers at 0.875
    rather than 0.95, which :func:`assert_coverage` catches. An efficiency
    comparison run without a coverage gate on both sides would report the broken
    interval as the better one.
    """
    rng = np.random.default_rng(37)
    n = 5
    draws = rng.normal(TRUTH, SIGMA, size=(REPS, n))
    means = draws.mean(axis=1)
    errors = draws.std(axis=1, ddof=1) / np.sqrt(n)

    def study(critical: float) -> MonteCarloResult:
        return MonteCarloResult(
            estimates=means,
            standard_errors=errors,
            covered=None,
            rejected=None,
            truth=TRUTH,
            lowers=means - critical * errors,
            uppers=means + critical * errors,
        )

    normal, student = study(Z_95), study(T_95_DF4)
    assert normal.mean_width / student.mean_width == pytest.approx(
        Z_95 / T_95_DF4, rel=1e-12
    )
    assert_narrower(normal, student, "1.96 against t at n=5")
    assert_coverage(student, 0.95, "t interval at n=5")
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_coverage(normal, 0.95, "1.96 interval at n=5")

    # The t interval is 1.33 times the oracle width and still not vacuous: the
    # study watches it miss at its nominal rate, which is what tells the two
    # apart.
    assert width_ratio(student, 0.95) == pytest.approx(1.33, rel=0.1)
    assert_intervals_informative(student, 0.95, "t interval at n=5")


def test_the_band_matches_the_textbook_binomial_interval():
    """The band is nominal +- sigmas * sqrt(p(1-p)/n), not something invented."""
    nominal, reps, sigmas = 0.95, 400, 3.0
    expected = sigmas * np.sqrt(nominal * (1 - nominal) / reps)
    low, high = binomial_band(nominal, reps, sigmas)
    assert low == pytest.approx(nominal - expected)
    assert high == pytest.approx(nominal + expected)
