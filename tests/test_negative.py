"""Negative tests: every gate must reject input that violates its property.

This is the file that makes the rest of the package worth anything. An assertion
helper is trivially satisfiable by an implementation that checks nothing, and a
suite built on such a helper reports success while testing nothing at all. So
each gate is exercised twice here: once on input that satisfies the property,
where it must stay silent, and once on input that violates it, where it must
raise.

The motivating case is real. The version of ``assert_rate`` this package was
extracted from took either a count or a rate and guessed which::

    observed = successes / reps if successes > 1 else float(successes)

One hit in four hundred replicates against a nominal 95% is a total failure of
the estimator. One is not greater than one, so it was read as a rate of 1.0,
which lies inside the band, and the assertion passed. ``test_a_count_of_one_is_
not_a_rate_of_one`` below is that exact case, and it is the reason counts and
rates are separate functions.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from simcheck import (
    MonteCarloResult,
    assert_count_rate,
    assert_coverage,
    assert_proportion,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
)


def _study(
    reps: int = 400,
    bias: float = 0.0,
    truth: float = 1.0,
    sd: float = 0.1,
    se_scale: float = 1.0,
    coverage: float | None = None,
    rejection: float = 0.05,
    seed: int = 0,
) -> MonteCarloResult:
    """Build a study with known properties.

    Parameters
    ----------
    reps
        Replicates.
    bias
        Constant added to every estimate.
    truth
        True value.
    sd
        Spread of the estimates.
    se_scale
        Multiplier on the reported standard error, so 1.0 is calibrated.
    coverage
        Exact fraction of replicates to mark covered. Defaults to 0.95.
    rejection
        Exact fraction of replicates to mark rejected.
    seed
        Generator seed.

    Returns
    -------
    MonteCarloResult
        The constructed study.
    """
    rng = np.random.default_rng(seed)
    estimates = rng.normal(truth + bias, sd, reps)
    errors = np.full(reps, sd * se_scale)
    hits = round((0.95 if coverage is None else coverage) * reps)
    covered = np.zeros(reps, dtype=bool)
    covered[:hits] = True
    rejected = np.zeros(reps, dtype=bool)
    rejected[: round(rejection * reps)] = True
    return MonteCarloResult(estimates, errors, covered, rejected, truth)


# --------------------------------------------------------------------------
# The bug this package exists to not repeat.
# --------------------------------------------------------------------------


def test_a_count_of_one_is_not_a_rate_of_one():
    """One hit in 400 must fail a nominal-95% check, not pass it.

    The extracted implementation read a count of 1 as a rate of 1.0 and passed.
    This is the regression test for the worst possible result being reported as
    the best possible one.
    """
    with pytest.raises(AssertionError):
        assert_count_rate(1, 400, 0.95, "catastrophic under-coverage")


def test_passing_a_count_to_the_rate_gate_is_rejected():
    """A count where a rate is expected must raise, not be guessed at."""
    with pytest.raises(ValueError, match="use assert_count_rate"):
        assert_proportion(380, 400, 0.95, "count passed as rate")


def test_a_count_above_the_replicate_count_is_rejected():
    """More successes than replicates is a caller error, not a rate."""
    with pytest.raises(ValueError, match=r"must be in \[0, 400\]"):
        assert_count_rate(401, 400, 0.95)


# --------------------------------------------------------------------------
# Each gate: silent when the property holds, raising when it does not.
# --------------------------------------------------------------------------


def test_assert_unbiased_passes_on_an_unbiased_estimator():
    """The gate must not fire on a correct estimator."""
    assert_unbiased(_study(bias=0.0), "unbiased")


def test_assert_unbiased_fails_on_a_biased_estimator():
    """A bias of five sampling sds cannot be missed."""
    with pytest.raises(AssertionError, match="standard errors from zero"):
        assert_unbiased(_study(bias=0.5, sd=0.1), "biased")


def test_assert_unbiased_resolves_smaller_bias_with_more_replicates():
    """The gate must tighten as the study grows, without any edit.

    This is the property that makes a replicate-derived tolerance worth having:
    the same call is lenient in a fast tier and strict in a deep one.
    """
    small = _study(reps=50, bias=0.02, sd=0.1)
    large = _study(reps=20000, bias=0.02, sd=0.1)

    assert_unbiased(small, "too small a study to resolve 0.02")
    with pytest.raises(AssertionError):
        assert_unbiased(large, "large enough to resolve 0.02")


def test_assert_coverage_passes_at_the_nominal_rate():
    """The gate must not fire on correctly calibrated intervals."""
    assert_coverage(_study(coverage=0.95), 0.95, "calibrated")


def test_assert_coverage_fails_on_under_coverage():
    """Intervals covering 80% of the time must fail a 95% claim."""
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_coverage(_study(coverage=0.80), 0.95, "under-covering")


def test_assert_coverage_fails_on_over_coverage():
    """Over-coverage is a defect too: the interval is wider than claimed."""
    with pytest.raises(AssertionError):
        assert_coverage(_study(reps=400, coverage=1.0), 0.95, "over-covering")


def test_assert_se_calibrated_passes_when_the_se_matches_the_spread():
    """The gate must not fire on an honest standard error."""
    assert_se_calibrated(_study(sd=0.1, se_scale=1.0), "calibrated", tolerance=0.15)


def test_assert_se_calibrated_fails_on_an_overconfident_standard_error():
    """Halving the reported standard error must be caught."""
    with pytest.raises(AssertionError, match="times the observed spread"):
        assert_se_calibrated(_study(sd=0.1, se_scale=0.5), "overconfident")


def test_assert_se_calibrated_fails_on_a_conservative_standard_error():
    """Inflation is a defect too -- it means the intervals are not 95%."""
    with pytest.raises(AssertionError):
        assert_se_calibrated(_study(sd=0.1, se_scale=2.0), "conservative")


def test_assert_se_calibrated_refuses_a_degenerate_study():
    """An estimator that never varies gives nothing to compare against.

    Silently passing here would certify a constant estimator as well calibrated.
    """
    constant = MonteCarloResult(
        estimates=np.full(100, 1.0),
        standard_errors=np.full(100, 0.1),
        covered=np.ones(100, dtype=bool),
        rejected=np.zeros(100, dtype=bool),
        truth=1.0,
    )
    with pytest.raises(AssertionError, match="did not vary"):
        assert_se_calibrated(constant, "constant estimator")


# --------------------------------------------------------------------------
# The band itself.
# --------------------------------------------------------------------------


def test_the_band_narrows_as_replicates_grow():
    """A tolerance that did not shrink with the study would be a constant."""
    widths = [
        binomial_band(0.95, reps)[1] - binomial_band(0.95, reps)[0]
        for reps in (100, 400, 1600, 6400)
    ]
    assert all(a > b for a, b in pairwise(widths)), widths
    # Four times the replicates should halve the width, since it goes as
    # 1/sqrt(reps). Asserting the rate, not merely the direction. Measured at a
    # nominal 0.5 rather than 0.95: at 0.95 and only 100 replicates the upper
    # bound clips at 1.0, which truncates the width and breaks the relationship.
    unclipped = [
        binomial_band(0.5, reps)[1] - binomial_band(0.5, reps)[0] for reps in (100, 400)
    ]
    assert unclipped[0] / unclipped[1] == pytest.approx(2.0, rel=1e-9)


def test_the_band_is_clipped_to_probabilities():
    """A nominal rate near one must not produce an upper bound above one."""
    low, high = binomial_band(0.99, 25)
    assert 0.0 <= low <= high <= 1.0


@pytest.mark.parametrize(
    ("nominal", "reps", "sigmas"),
    [(1.5, 100, 3.0), (-0.1, 100, 3.0), (0.95, 0, 3.0), (0.95, 100, -1.0)],
)
def test_the_band_rejects_impossible_arguments(nominal, reps, sigmas):
    """Nonsense in must not silently produce a band."""
    with pytest.raises(ValueError, match="must be"):
        binomial_band(nominal, reps, sigmas)


# --------------------------------------------------------------------------
# The result object.
# --------------------------------------------------------------------------


def test_ragged_arrays_are_rejected():
    """Mismatched per-replicate arrays mean the study is not what it claims."""
    with pytest.raises(ValueError, match="disagree in length"):
        MonteCarloResult(
            estimates=np.zeros(10),
            standard_errors=np.zeros(9),
            covered=np.zeros(10, dtype=bool),
            rejected=np.zeros(10, dtype=bool),
            truth=0.0,
        )


def test_an_empty_study_is_rejected():
    """Zero replicates would make every rate a division by zero."""
    with pytest.raises(ValueError, match="at least one replicate"):
        MonteCarloResult(
            estimates=np.zeros(0),
            standard_errors=np.zeros(0),
            covered=np.zeros(0, dtype=bool),
            rejected=np.zeros(0, dtype=bool),
            truth=0.0,
        )
