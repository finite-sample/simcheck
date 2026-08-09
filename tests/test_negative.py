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

import math
import subprocess
import sys
import textwrap
from itertools import pairwise

import numpy as np
import pytest

from simcheck import (
    MonteCarloResult,
    assert_count_rate,
    assert_coverage,
    assert_intervals_informative,
    assert_more_powerful,
    assert_narrower,
    assert_power,
    assert_proportion,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
    se_ratio_tolerance,
    vacuous_width_ratio,
    width_ratio,
)

# Two-sided normal quantile for 95%, from tables. simcheck has no scipy
# dependency and should not gain one so a test can call `norm.ppf`.
Z_95 = 1.959964


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


def _interval_study(
    reps: int = 400,
    sd: float = 0.1,
    truth: float = 1.0,
    bias: float = 0.0,
    half_width: float | None = None,
    width_cv: float = 0.0,
    seed: int = 0,
) -> MonteCarloResult:
    """Build a study whose intervals have a known width and a known coverage.

    Parameters
    ----------
    reps
        Replicates.
    sd
        Spread of the estimates.
    truth
        True value.
    bias
        Constant added to every estimate.
    half_width
        Half-width of every interval. Defaults to the calibrated 95% one,
        ``1.96 * sd``, which makes ``width_ratio`` exactly one.
    width_cv
        Relative spread of the width across replicates. Zero gives every
        replicate the same width, as a procedure with a known scale would; a
        real procedure estimates the scale and so has a width that varies.
    seed
        Generator seed.

    Returns
    -------
    MonteCarloResult
        The constructed study. ``covered`` is left for the result object to
        derive from the endpoints, which is also what exercises that path.
    """
    rng = np.random.default_rng(seed)
    estimates = rng.normal(truth + bias, sd, reps)
    half = Z_95 * sd if half_width is None else half_width
    if width_cv:
        half = half * np.abs(1.0 + width_cv * rng.standard_normal(reps))
    return MonteCarloResult(
        estimates=estimates,
        standard_errors=np.full(reps, sd),
        covered=None,
        rejected=None,
        truth=truth,
        lowers=estimates - half,
        uppers=estimates + half,
    )


def _decision_study(rejection: float, reps: int = 400) -> MonteCarloResult:
    """A study with an exact rejection rate and nothing else of interest.

    Parameters
    ----------
    rejection
        Exact fraction of replicates to mark rejected.
    reps
        Replicates.

    Returns
    -------
    MonteCarloResult
        The constructed study.
    """
    rejected = np.zeros(reps, dtype=bool)
    rejected[: round(rejection * reps)] = True
    return MonteCarloResult(
        estimates=np.linspace(0.9, 1.1, reps),
        standard_errors=np.full(reps, 0.05),
        covered=None,
        rejected=rejected,
        truth=1.0,
    )


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


def test_the_se_tolerance_narrows_as_the_study_grows():
    """The derived tolerance must be a function of the replicate count.

    A constant 0.15 is simultaneously too loose to catch a 12% error in a deep
    study and tight enough to fail a correct estimator in a shallow one. The
    derived band is 0.21 at 100 replicates and 0.05 at 2000.
    """
    tolerances = [
        se_ratio_tolerance(_study(reps=reps)) for reps in (100, 400, 2000, 20000)
    ]
    assert all(a > b for a, b in pairwise(tolerances)), tolerances
    # Four times the replicates halves the band, since the spread of a sample
    # standard deviation goes as 1/sqrt(2(reps-1)).
    assert tolerances[0] / tolerances[2] == pytest.approx(4.47, rel=0.02)


def test_the_same_se_error_passes_a_shallow_study_and_fails_a_deep_one():
    """The point of deriving the tolerance, in one test.

    A reported standard error 12% too large sits inside the hand-picked 0.15 at
    every replicate count -- a 20000-replicate study could resolve it to better
    than 1% and was told not to look. Derived, the same call passes at 100
    replicates, where the study genuinely cannot tell, and fails at 2000.
    """
    assert_se_calibrated(_study(reps=100, se_scale=1.12), "too small to resolve 12%")
    with pytest.raises(AssertionError, match="times the observed spread"):
        assert_se_calibrated(_study(reps=2000, se_scale=1.12), "deep enough")
    # And the hand-picked constant would have passed both.
    assert_se_calibrated(_study(reps=2000, se_scale=1.12), "0.15", tolerance=0.15)


def test_assert_se_calibrated_says_so_when_no_standard_error_was_reported():
    """A NaN standard error is an absent measurement, not a zero spread.

    ``Estimate`` documents that leaving ``standard_error`` as NaN means the gate
    "will have nothing to check and will say so". It used to say the estimator
    did not vary, which is a different -- and false -- diagnosis.
    """
    study = MonteCarloResult(
        estimates=np.linspace(0.9, 1.1, 100),
        standard_errors=np.full(100, np.nan),
        covered=None,
        rejected=None,
        truth=1.0,
    )
    with pytest.raises(AssertionError, match="reported no usable standard error"):
        assert_se_calibrated(study, "no standard error")


# --------------------------------------------------------------------------
# Vacuity: an interval so wide it cannot fail.
# --------------------------------------------------------------------------


def test_assert_intervals_informative_passes_a_calibrated_interval():
    """A 1.96-sigma interval is exactly as wide as its level requires."""
    study = _interval_study()
    assert width_ratio(study) == pytest.approx(1.0, rel=0.05)
    assert_intervals_informative(study, 0.95, "calibrated")


def test_assert_intervals_informative_fails_an_interval_that_cannot_miss():
    """The defect `assert_coverage` cannot see: coverage of 1.0 by construction."""
    study = _interval_study(half_width=10 * 0.1)
    with pytest.raises(AssertionError, match="intervals are vacuous"):
        assert_intervals_informative(study, 0.95, "ten-sigma interval")


def test_a_conservative_but_honest_interval_is_not_called_vacuous():
    """The false-positive guard, and the reason the gate has two conditions.

    A Student t interval at n=5 is 1.33 times the width the normal oracle needs,
    which is above the naive width threshold at any replicate count -- and it is
    correct. It misses at its nominal rate, the study sees those misses, so it is
    conservative rather than vacuous. A gate that fired here would be switched
    off within a week.
    """
    reps, n = 2000, 5
    rng = np.random.default_rng(4)
    draws = rng.normal(1.0, 1.0, size=(reps, n))
    means = draws.mean(axis=1)
    errors = draws.std(axis=1, ddof=1) / np.sqrt(n)
    half = 2.776445 * errors  # t_{0.975, 4}, from tables.
    study = MonteCarloResult(
        estimates=means,
        standard_errors=errors,
        covered=None,
        rejected=None,
        truth=1.0,
        lowers=means - half,
        uppers=means + half,
    )
    assert width_ratio(study) == pytest.approx(1.33, rel=0.1)
    assert width_ratio(study) > vacuous_width_ratio(0.95, reps) / 2
    assert_coverage(study, 0.95, "t interval at n=5")
    assert_intervals_informative(study, 0.95, "t interval at n=5")


def test_width_alone_does_not_make_an_interval_vacuous():
    """A wide interval the study watched fail is a different defect.

    Here the estimator is biased by five standard deviations and its intervals
    are three times as wide as they need to be, so it misses about a fifth of the
    time. That is a bias, which `assert_unbiased` reports; calling it vacuity as
    well would send the reader after the wrong thing.
    """
    study = _interval_study(bias=0.5, half_width=3 * Z_95 * 0.1)
    assert width_ratio(study) == pytest.approx(3.0, rel=0.05)
    assert_intervals_informative(study, 0.95, "wide, biased, and caught")
    with pytest.raises(AssertionError, match="standard errors from zero"):
        assert_unbiased(study, "wide, biased, and caught")


@pytest.mark.parametrize(
    ("case", "half_width", "reps"),
    [
        # geoinference: "an interval so wide it covers every single time, which
        # is a defect the assertion cannot see".
        ("geoinference: coverage 1.0 at 300 replicates", 8 * 0.1, 300),
        # alsgls: "an interval covering 100% of the time, which means it is far
        # too wide, passed it".
        ("alsgls: pooled points, coverage 1.0", 6 * 0.1, 300),
        # incline: "an inflation heuristic drove this ratio to 3e7 while
        # coverage stayed high, because a vacuous interval covers everything".
        ("incline: reported se 3e7 times the error", 3e7 * Z_95 * 0.1, 400),
    ],
)
def test_the_three_workarounds_in_the_wild_are_caught(case, half_width, reps):
    """Each sibling repo hand-rolled a comment where this gate should have been.

    Three repositories independently hit the same hole in `assert_coverage` and
    wrote a comment about it instead of a test. If the gate cannot reproduce all
    three failures it is the wrong gate, so all three run here.
    """
    study = _interval_study(reps=reps, half_width=half_width)
    assert study.coverage == 1.0, case
    with pytest.raises(AssertionError, match="intervals are vacuous"):
        assert_intervals_informative(study, 0.95, case)


def test_the_vacuity_gate_needs_endpoints():
    """Coverage alone cannot answer the question, and saying so beats guessing."""
    with pytest.raises(ValueError, match="recorded no interval endpoints"):
        assert_intervals_informative(_study(), 0.95, "no endpoints")


def test_the_vacuity_threshold_comes_from_the_replicate_count():
    """More replicates resolve rarer misses, so the width limit rises with reps."""
    ratios = [vacuous_width_ratio(0.95, reps) for reps in (100, 400, 2000, 10000)]
    assert all(a < b for a, b in pairwise(ratios)), ratios
    # And it is the width whose implied miss rate leaves `alpha` misses expected
    # in the whole study: reps * 2 * (1 - Phi(r * z)) == alpha.
    reps = 400
    implied = math.erfc(ratios[1] * Z_95 / math.sqrt(2.0))
    assert reps * implied == pytest.approx(0.05, rel=1e-6)


# --------------------------------------------------------------------------
# Widths across two methods.
# --------------------------------------------------------------------------


def test_assert_narrower_passes_when_one_method_really_is_narrower():
    """The efficiency comparison, when the gap is real."""
    tight = _interval_study(half_width=Z_95 * 0.1, width_cv=0.2, seed=1)
    loose = _interval_study(half_width=1.5 * Z_95 * 0.1, width_cv=0.2, seed=2)
    assert_narrower(tight, loose, "tight against loose")


def test_assert_narrower_fails_when_the_gap_is_noise():
    """Two methods of the same width must not be ranked by the seed.

    ``a.mean_width < b.mean_width`` is true of one of the two whatever they are,
    which is exactly the assertion this replaces.
    """
    first = _interval_study(half_width=Z_95 * 0.1, width_cv=0.2, seed=1)
    second = _interval_study(half_width=Z_95 * 0.1001, width_cv=0.2, seed=1)
    assert first.mean_width < second.mean_width
    with pytest.raises(AssertionError, match="not measurably below"):
        assert_narrower(first, second, "a hair narrower")


def test_a_width_that_never_varies_needs_only_to_be_smaller():
    """With a known scale the width is not random, so any gap is a real gap.

    Both studies here produce exactly one width, so the Monte Carlo standard
    error of the difference is exactly zero and the gate reduces to a strict
    inequality. Demanding a multiple of a zero standard error would make the gate
    unsatisfiable for the one case where the answer is certain.
    """

    def fixed(width: float) -> MonteCarloResult:
        reps = 200
        return MonteCarloResult(
            estimates=np.linspace(0.4, 0.6, reps),
            standard_errors=np.full(reps, 0.05),
            covered=None,
            rejected=None,
            truth=0.5,
            lowers=np.zeros(reps),
            uppers=np.full(reps, width),
        )

    assert_narrower(fixed(0.19), fixed(0.20), "known scale")
    with pytest.raises(AssertionError, match="exactly zero Monte Carlo"):
        assert_narrower(fixed(0.20), fixed(0.19), "known scale, backwards")


def test_assert_narrower_fails_when_the_arguments_are_the_wrong_way_round():
    """A wired-up-backwards comparison must not pass."""
    tight = _interval_study(half_width=Z_95 * 0.1, seed=1)
    loose = _interval_study(half_width=2 * Z_95 * 0.1, seed=2)
    with pytest.raises(AssertionError, match="not measurably below"):
        assert_narrower(loose, tight, "backwards")


def test_assert_narrower_needs_endpoints_on_both_studies():
    """Comparing a width against an unmeasured one is not a comparison."""
    with pytest.raises(ValueError, match="`wide` study recorded no interval"):
        assert_narrower(_interval_study(), _study(), "one side missing")


# --------------------------------------------------------------------------
# Power.
# --------------------------------------------------------------------------


def test_assert_power_passes_a_test_that_meets_its_claim():
    """A test rejecting 80% of the time meets a claim of 0.75."""
    assert_power(_decision_study(0.80), 0.75, "adequately powered")


def test_assert_power_fails_an_underpowered_test():
    """Half the claimed power over 400 replicates cannot be missed."""
    with pytest.raises(AssertionError, match="below the one-sided"):
        assert_power(_decision_study(0.40), 0.80, "underpowered")


def test_assert_power_is_one_sided():
    """Rejecting more often than promised is not a defect of the test.

    ``assert_proportion`` bands both sides, which is right for size and wrong for
    power: it would fail a test for being better than claimed.
    """
    strong = _decision_study(0.95)
    assert_power(strong, 0.50, "better than claimed")
    with pytest.raises(AssertionError, match="outside the 3-sigma band"):
        assert_proportion(strong.rejection_rate, strong.reps, 0.50, "two-sided")


def test_assert_power_tightens_with_replicates():
    """The floor is a binomial band, so it closes on the claim as reps grow."""
    assert_power(_decision_study(0.74, reps=100), 0.80, "cannot resolve 6 points")
    with pytest.raises(AssertionError):
        assert_power(_decision_study(0.74, reps=4000), 0.80, "can resolve 6 points")


def test_assert_power_needs_decisions():
    """A study that recorded no rejections has no power, rather than zero power."""
    with pytest.raises(ValueError, match="recorded no reject/accept decisions"):
        assert_power(_interval_study(), 0.8, "no decisions")


def test_assert_power_rejects_an_impossible_claim():
    """A power outside [0, 1] is a caller error."""
    with pytest.raises(ValueError, match="must be a probability"):
        assert_power(_decision_study(0.5), 1.4)


def test_assert_more_powerful_passes_on_a_real_gap():
    """Half again the rejection rate over 400 replicates is not noise."""
    assert_more_powerful(_decision_study(0.80), _decision_study(0.50), "real gap")


def test_assert_more_powerful_fails_on_a_gap_that_is_noise():
    """Two points of rejection rate at 400 replicates is a coin flip.

    This is the assertion the gate replaces: ``strong > weak`` is satisfied by
    whichever method the seed favoured, and reports it as a finding.
    """
    strong, weak = _decision_study(0.52), _decision_study(0.50)
    assert strong.rejection_rate > weak.rejection_rate
    with pytest.raises(AssertionError, match="not measurably above"):
        assert_more_powerful(strong, weak, "two points of rate")


def test_assert_more_powerful_fails_when_the_arguments_are_the_wrong_way_round():
    """A comparison wired up backwards must not pass."""
    with pytest.raises(AssertionError, match="not measurably above"):
        assert_more_powerful(_decision_study(0.30), _decision_study(0.80), "backwards")


def test_assert_more_powerful_needs_decisions_on_both_studies():
    """One side without decisions is not a comparison."""
    with pytest.raises(ValueError, match="`less` study recorded no reject"):
        assert_more_powerful(_decision_study(0.8), _interval_study(), "one side")


# --------------------------------------------------------------------------
# Interval endpoints on the result object.
# --------------------------------------------------------------------------


def test_one_endpoint_without_the_other_is_rejected():
    """Half an interval is not an interval."""
    with pytest.raises(ValueError, match="needs both endpoints"):
        MonteCarloResult(
            estimates=np.zeros(10),
            standard_errors=np.ones(10),
            covered=None,
            rejected=None,
            truth=0.0,
            lowers=np.full(10, -1.0),
        )


def test_a_backwards_interval_is_rejected():
    """A negative width is not a width."""
    with pytest.raises(ValueError, match="upper endpoint below the lower"):
        MonteCarloResult(
            estimates=np.zeros(10),
            standard_errors=np.ones(10),
            covered=None,
            rejected=None,
            truth=0.0,
            lowers=np.full(10, 1.0),
            uppers=np.full(10, -1.0),
        )


def test_covered_contradicting_the_endpoints_is_rejected():
    """Two measurements of the same thing that disagree are one measurement too many."""
    with pytest.raises(ValueError, match="disagrees with the endpoints"):
        MonteCarloResult(
            estimates=np.zeros(10),
            standard_errors=np.ones(10),
            covered=np.zeros(10, dtype=bool),
            rejected=None,
            truth=0.0,
            lowers=np.full(10, -1.0),
            uppers=np.full(10, 1.0),
        )


def test_covered_is_filled_in_from_the_endpoints():
    """Endpoints determine coverage exactly, so a study with them has a rate."""
    study = _interval_study(reps=100, half_width=Z_95 * 0.1)
    assert study.covered is not None
    assert 0.85 <= study.coverage <= 1.0


def test_a_study_without_endpoints_has_no_widths():
    """Zero would read as an infinitely precise estimator, not an unmeasured one."""
    with pytest.raises(ValueError, match="recorded no interval endpoints"):
        _ = _study().mean_width


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


def test_the_gates_still_fire_under_python_o():
    """`python -O` must not turn every gate into a no-op.

    ``assert`` statements are removed entirely under ``-O``. A package whose
    entire product is assertions would then pass everything silently -- the exact
    failure mode it exists to prevent, in itself. This was real: the first
    version of the gates used bare ``assert`` and, under ``-O``, certified one
    hit in four hundred replicates as acceptable 95% coverage.

    Run in a subprocess because ``-O`` is a interpreter-startup flag; the test
    session itself cannot toggle it.
    """
    program = textwrap.dedent("""
        import sys
        from simcheck import (
            MonteCarloResult, assert_count_rate, assert_coverage,
            assert_intervals_informative, assert_more_powerful, assert_narrower,
            assert_power, assert_proportion, assert_se_calibrated,
            assert_unbiased,
        )
        import numpy as np

        assert __debug__ is False, "subprocess is not running under -O"

        fired = []
        def check(name, fn):
            try:
                fn()
            except AssertionError:
                fired.append(name)

        check("count_rate", lambda: assert_count_rate(1, 400, 0.95))
        check("proportion", lambda: assert_proportion(0.10, 400, 0.95))

        biased = MonteCarloResult(
            estimates=np.linspace(5.0, 5.2, 400),
            # A quarter of the true spread, so se_calibrated is unambiguously
            # violated rather than sitting near its tolerance.
            standard_errors=np.full(400, 0.015),
            covered=np.zeros(400, dtype=bool),
            rejected=np.zeros(400, dtype=bool),
            truth=1.0,
        )
        check("unbiased", lambda: assert_unbiased(biased))
        check("coverage", lambda: assert_coverage(biased, 0.95))
        check("se_calibrated", lambda: assert_se_calibrated(biased))

        estimates = np.linspace(0.9, 1.1, 400)
        def interval(half, rejection):
            flags = np.zeros(400, dtype=bool)
            flags[: round(rejection * 400)] = True
            return MonteCarloResult(
                estimates=estimates,
                standard_errors=np.full(400, 0.058),
                covered=None,
                rejected=flags,
                truth=1.0,
                lowers=estimates - half,
                uppers=estimates + half,
            )

        vacuous = interval(1.0, 0.5)
        check("intervals_informative",
              lambda: assert_intervals_informative(vacuous, 0.95))
        check("narrower", lambda: assert_narrower(vacuous, interval(0.2, 0.5)))
        check("power", lambda: assert_power(interval(1.0, 0.30), 0.80))
        check("more_powerful",
              lambda: assert_more_powerful(interval(1.0, 0.50), interval(1.0, 0.49)))

        print(",".join(sorted(fired)))
    """)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-O", "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    fired = set(result.stdout.strip().split(","))
    assert fired == {
        "count_rate",
        "coverage",
        "intervals_informative",
        "more_powerful",
        "narrower",
        "power",
        "proportion",
        "se_calibrated",
        "unbiased",
    }, f"under -O only these gates fired: {sorted(fired)}"
