"""Assertions whose tolerance comes from the replicate count.

An assertion like ``coverage > 0.85`` for a nominal 95% interval is arbitrary:
far too loose at ten thousand replicates, and tight enough to fail spuriously at
fifty. Every gate here derives its tolerance from the number of replicates
instead, so the same line of test code loosens automatically in a fast tier and
tightens in a deep one, and says in its failure message how far outside the band
the result fell.

Counts and rates are separate functions on purpose. The version this was
extracted from took either, and guessed which it had been given::

    observed = successes / reps if successes > 1 else float(successes)

For a nominal 95% coverage study, one hit in four hundred replicates -- a total
failure of the estimator -- is not greater than one, so it was read as a *rate*
of 1.0, which sits inside the band, and the test passed. The worst possible
result was reported as the best possible one. There is no heuristic that
distinguishes a count of 1 from a rate of 1.0; the caller has to say which.

Every gate raises ``AssertionError`` explicitly rather than using an ``assert``
statement. ``assert`` is removed entirely by ``python -O``, so a package whose
whole product is assertions would silently pass everything under optimisation --
the exact failure mode it exists to prevent, in itself. ``test_negative.py``
runs the gates in an ``-O`` subprocess to keep it that way.

**Where each threshold comes from is written down.** Three shapes appear here:

* A rate against a claimed rate -- coverage, size, power. The band is binomial in
  ``reps`` and there is nothing to choose: :func:`binomial_band`.
* A quantity against its own Monte Carlo standard error -- bias, the
  claimed-to-actual standard error ratio, the gap between two methods. The
  tolerance is again ``reps``, through the sampling distribution of the quantity:
  :func:`se_ratio_tolerance`.
* An interval's *width*. Here the reference width is derived from the study
  (:func:`vacuous_width_ratio`), but the multiple of it at which conservatism
  becomes vacuity is a judgement, and that function's docstring says so plainly
  rather than presenting a chosen number as a derived one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .results import MonteCarloResult

__all__ = [
    "GATE_SIGMAS",
    "assert_count_rate",
    "assert_coverage",
    "assert_intervals_informative",
    "assert_more_powerful",
    "assert_narrower",
    "assert_power",
    "assert_proportion",
    "assert_se_calibrated",
    "assert_unbiased",
    "binomial_band",
    "se_ratio_tolerance",
    "vacuous_width_ratio",
    "width_ratio",
]

# How many standard errors a result may sit from nominal before it counts as
# miscalibrated. Three is loose enough that a correct estimator essentially never
# trips it -- roughly one false failure in 370 -- and tight enough to catch a 20%
# error in a reported standard error at a few hundred replicates.
GATE_SIGMAS = 3.0


def binomial_band(
    nominal: float, reps: int, sigmas: float = GATE_SIGMAS
) -> tuple[float, float]:
    """The interval a well-calibrated rate should land in.

    Args:
        nominal: The rate the estimator claims, for instance 0.95 for coverage.
        reps: Number of replicates.
        sigmas: How many binomial standard errors of slack to allow.

    Returns:
        tuple of float: Lower and upper bounds, clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``nominal`` is outside ``[0, 1]``, ``reps`` is not positive,
            or ``sigmas`` is negative.

    Examples:
    >>> low, high = binomial_band(0.95, 400)
    >>> round(low, 4), round(high, 4)
    (0.9173, 0.9827)
    """
    if not 0.0 <= nominal <= 1.0:
        raise ValueError(f"nominal must be a probability, got {nominal}")
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    if sigmas < 0:
        raise ValueError(f"sigmas must be non-negative, got {sigmas}")
    spread = sigmas * float(np.sqrt(nominal * (1.0 - nominal) / reps))
    return max(0.0, nominal - spread), min(1.0, nominal + spread)


def _band_failure(
    observed: float, reps: int, nominal: float, sigmas: float
) -> str | None:
    """Describe how an observed rate misses its band, or None if it does not.

    Reporting rather than raising is what lets every public gate below raise
    AssertionError itself. A gate that delegated its raise to another function
    would still be correct, but its documented exceptions could no longer be
    checked against its body -- and the tooling that catches a docstring drifting
    from its code would go quiet on exactly the functions whose whole job is
    raising.

    Args:
        observed: The observed rate.
        reps: Replicates it was computed over.
        nominal: The claimed rate.
        sigmas: Slack, in binomial standard errors.

    Returns:
        str | None: The failure message, or None when the rate is inside the band.
    """
    low, high = binomial_band(nominal, reps, sigmas)
    if low <= observed <= high:
        return None
    return (
        f"observed rate {observed:.4f} outside the {sigmas:g}-sigma "
        f"band [{low:.4f}, {high:.4f}] for a nominal {nominal:.4f} "
        f"over {reps} replicates"
    )


def _normal_tail_quantile(tail: float) -> float:
    """The ``z`` with ``P(Z > z) == tail`` for a standard normal ``Z``.

    numpy has no normal quantile function and simcheck depends on numpy alone, so
    this inverts ``math.erfc`` by bisection rather than acquiring scipy for one
    call. It works in the upper tail rather than in the cumulative probability
    because the tail is where it is used: ``1 - Phi(z)`` has lost every
    significant digit by ``z = 9``, while ``erfc`` has not.

    Args:
        tail: Upper-tail probability, in ``(0, 0.5]``.

    Returns:
        float: The quantile.

    Raises:
        ValueError: If ``tail`` is outside ``(0, 0.5]``.
    """
    if not 0.0 < tail <= 0.5:
        raise ValueError(f"tail probability must be in (0, 0.5], got {tail}")
    low, high = 0.0, 40.0
    for _ in range(120):
        mid = 0.5 * (low + high)
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > tail:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def vacuous_width_ratio(nominal: float, reps: int) -> float:
    """How many times the calibrated width an interval may reach before it is vacuous.

    **The reference width is derived, and contains no chosen number.** For an
    estimator whose sampling distribution is approximately normal with spread
    ``sigma``, the shortest interval that contains the truth at rate ``1 - alpha``
    has width ``2 * z_{1 - alpha/2} * sigma``. The study measures ``sigma``
    itself, as ``sampling_sd``, so the width an interval *should* have is a
    measurement rather than a threshold. :func:`width_ratio` reports the observed
    mean width in units of it.

    **The multiple returned here is derived from ``reps``, with one convention in
    it that is stated rather than hidden.** Widen a calibrated interval by a
    factor ``r`` and its miss rate falls to ``q(r) = 2 * (1 - Phi(r * z))``. This
    function returns the ``r`` at which the *whole study* expects fewer than
    ``alpha`` misses -- that is, at which observing a single failure would take
    ``1 / alpha`` studies of this size. Past that point the coverage a study
    reports is a property of the width rather than of the estimator: the interval
    could not have failed, so its covering says nothing. Solving
    ``reps * q(r) = alpha`` gives ``r = z_{1 - alpha/(2*reps)} / z_{1 - alpha/2}``.

    The threshold therefore *loosens* as the study grows -- 1.78 at 100
    replicates, 1.96 at 400, 2.15 at 2000 for a nominal 0.95 -- which inverts the
    usual direction and is meant to: more replicates resolve rarer failures, so
    an interval must be wider before a study of that size can no longer see it
    fail.

    **What is not derived.** How many expected misses per study counts as "could
    not have failed" -- ``alpha`` of one, here, taken from the interval's own
    claim rather than invented -- is a convention. No sampling distribution fixes
    it, because correct procedures occupy the whole range above one: a Student t
    interval at ``n = 5`` is 1.33 times the normal oracle width and an
    anytime-valid interval is wider still, both of them right. That is why
    :func:`assert_intervals_informative` does not fail on width alone, but only
    on width *together with* a study that never once saw the interval miss. A
    procedure that fails at its nominal rate is not vacuous however wide it is,
    and no threshold on width can be asked to know that.

    Args:
        nominal: The level the intervals claim, for instance 0.95.
        reps: Number of replicates in the study.

    Returns:
        float: The width multiple at which the study loses the ability to
        observe the interval failing.

    Raises:
        ValueError: If ``nominal`` is not strictly inside ``(0, 1)`` or ``reps``
            is not positive.

    Examples:
    >>> round(vacuous_width_ratio(0.95, 400), 3)
    1.957
    >>> round(vacuous_width_ratio(0.95, 100), 3)
    1.776
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal must be strictly inside (0, 1), got {nominal}")
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    alpha = 1.0 - nominal
    return _normal_tail_quantile(alpha / (2.0 * reps)) / _normal_tail_quantile(
        alpha / 2.0
    )


def width_ratio(result: MonteCarloResult, nominal: float = 0.95) -> float:
    """Mean interval width over the width a calibrated interval would have.

    One means the interval is as narrow as its level allows against the spread
    this estimator actually has; two means it is twice as wide as it needs to be.
    The denominator, ``2 * z_{1 - alpha/2} * sampling_sd``, comes from the study,
    so no absolute width is written down anywhere.

    Args:
        result: A completed Monte Carlo study that recorded interval endpoints.
        nominal: The level the intervals claim.

    Returns:
        float: The ratio.

    Raises:
        ValueError: If the study recorded no interval endpoints, if ``nominal``
            is not strictly inside ``(0, 1)``, or if the estimator did not vary
            across replicates, which leaves no spread to measure the width
            against.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal must be strictly inside (0, 1), got {nominal}")
    spread = result.sampling_sd
    if not spread:
        raise ValueError(
            f"the estimator did not vary across {result.reps} replicates, so "
            "there is no sampling spread to compare its interval width against"
        )
    calibrated = 2.0 * _normal_tail_quantile((1.0 - nominal) / 2.0) * spread
    return result.mean_width / calibrated


def _mean_gap_se(first: np.ndarray, second: np.ndarray) -> float:
    """Monte Carlo standard error of the difference between two means.

    The two studies are treated as independent, which is what they are when they
    were run under different seeds. Running them under the *same* seed pairs the
    replicates and makes the true standard error of the difference smaller, so
    this is the conservative choice: a paired comparison passing this gate would
    also pass a paired one.

    Args:
        first: Per-replicate values from one study.
        second: Per-replicate values from the other.

    Returns:
        float: The standard error of ``mean(first) - mean(second)``.
    """
    if len(first) < 2 or len(second) < 2:
        return 0.0
    return math.sqrt(
        float(np.var(first, ddof=1)) / len(first)
        + float(np.var(second, ddof=1)) / len(second)
    )


def _gap_is_unresolved(gap: float, standard_error: float, sigmas: float) -> bool:
    """Whether a difference is too small for the studies to resolve.

    Args:
        gap: The observed difference, signed so that positive is the claim.
        standard_error: Monte Carlo standard error of the difference.
        sigmas: How many of them the gap must exceed.

    Returns:
        bool: True when the claim is not established.
    """
    if standard_error == 0.0:
        return gap <= 0.0
    return gap <= sigmas * standard_error


def assert_proportion(
    observed: float,
    reps: int,
    nominal: float,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if an observed **rate** is inconsistent with the claimed one.

    Args:
        observed: The observed rate, in ``[0, 1]``. Pass a count to
            :func:`assert_count_rate` instead.
        reps: Number of replicates the rate was computed over.
        nominal: The claimed rate.
        label: Included in the failure message.
        sigmas: Slack, in binomial standard errors.

    Raises:
        ValueError: If ``observed`` is not in ``[0, 1]``, which usually means a
            count was passed where a rate was expected.
        AssertionError: If the observed rate falls outside the band.
    """
    if not 0.0 <= observed <= 1.0:
        raise ValueError(
            f"observed must be a rate in [0, 1], got {observed}. If this is a "
            "count of successes, use assert_count_rate."
        )
    problem = _band_failure(observed, reps, nominal, sigmas)
    if problem is not None:
        raise AssertionError(f"{label}: {problem}")


def assert_count_rate(
    successes: int,
    reps: int,
    nominal: float,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if a **count** of successes is inconsistent with the claimed rate.

    Args:
        successes: Number of successes, in ``[0, reps]``.
        reps: Number of replicates.
        nominal: The claimed rate.
        label: Included in the failure message.
        sigmas: Slack, in binomial standard errors.

    Raises:
        ValueError: If ``successes`` is negative or exceeds ``reps``.
        AssertionError: If the implied rate falls outside the band.
    """
    if not 0 <= successes <= reps:
        raise ValueError(f"successes must be in [0, {reps}], got {successes}")
    problem = _band_failure(successes / reps, reps, nominal, sigmas)
    if problem is not None:
        raise AssertionError(f"{label}: {problem}")


def assert_unbiased(
    result: MonteCarloResult, label: str = "", sigmas: float = GATE_SIGMAS
) -> None:
    """Fail if the estimator's mean is distinguishable from the truth.

    The comparison is against the *Monte Carlo* standard error of the mean, so a
    bias too small for the study to resolve does not fail, and the study can be
    made more demanding simply by running more replicates.

    Args:
        result: A completed Monte Carlo study.
        label: Included in the failure message.
        sigmas: How many Monte Carlo standard errors of slack to allow.

    Raises:
        AssertionError: If the bias t statistic exceeds the gate.
    """
    if not abs(result.bias_t) < sigmas:
        raise AssertionError(
            f"{label}: bias {result.bias:+.6f} is {result.bias_t:+.2f} Monte "
            f"Carlo standard errors from zero over {result.reps} replicates "
            f"(sampling sd {result.sampling_sd:.6f}, mc se {result.mc_se:.6f})"
        )


def assert_coverage(
    result: MonteCarloResult,
    nominal: float = 0.95,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if interval coverage is inconsistent with the nominal level.

    Args:
        result: A completed Monte Carlo study.
        nominal: The level the intervals claim.
        label: Included in the failure message.
        sigmas: Slack, in binomial standard errors.

    Raises:
        AssertionError: If coverage falls outside the band.
        ValueError: If the study recorded no intervals, so there is no coverage
            rate to test.
    """
    if result.covered is None:
        raise ValueError(
            f"{label or 'this study'} recorded no confidence intervals, so its "
            "coverage cannot be checked. Have the estimator report `lower` and "
            "`upper` on every replicate."
        )
    problem = _band_failure(result.coverage, result.reps, nominal, sigmas)
    if problem is not None:
        raise AssertionError(f"{f'{label} coverage'.strip()}: {problem}")


def se_ratio_tolerance(result: MonteCarloResult, sigmas: float = GATE_SIGMAS) -> float:
    """How far ``se_ratio`` can sit from one on Monte Carlo noise alone.

    ``se_ratio`` is ``mean(reported standard errors) / sd(estimates)``, and both
    halves are estimated from the same ``reps`` replicates, so it is noisy even
    when the estimator is perfect. Its sampling distribution is available:

    * The numerator is a mean of ``reps`` reported standard errors, so its
      relative standard error is ``cv / sqrt(reps)``, where ``cv`` is their
      coefficient of variation across replicates. An estimator that reports the
      same standard error every time contributes nothing here.
    * The denominator is a sample standard deviation of ``reps`` draws, whose
      relative standard error is ``1 / sqrt(2 * (reps - 1))`` -- exactly for
      normal estimates and closely for anything with a finite fourth moment.

    Adding them in quadrature and multiplying by ``sigmas`` gives the band. The
    two are in fact positively correlated for most estimators -- a replicate that
    produces a large estimate often reports a large standard error too -- and
    ignoring that overstates the variance, which makes this the lenient choice.

    At 100 replicates the band is about 0.21 and at 2000 about 0.05, so the same
    call is a sanity check in a fast tier and a real test in a deep one. That is
    the whole point: 0.15 was the one number in this package chosen by hand
    rather than derived, and it was simultaneously too loose to catch a 12%
    error in a 2000-replicate study and tight enough to fail a correct estimator
    roughly one time in ten at 50.

    Args:
        result: A completed Monte Carlo study.
        sigmas: How many Monte Carlo standard errors of slack to allow.

    Returns:
        float: The largest deviation of ``se_ratio`` from one that this study
        cannot distinguish from noise.

    Raises:
        ValueError: If the study has fewer than two replicates, which leaves the
            spread -- and so the ratio -- undefined.
    """
    if result.reps < 2:
        raise ValueError(
            "a single replicate has no spread, so its reported standard error "
            "cannot be checked against anything"
        )
    reported = np.asarray(result.standard_errors, dtype=float)
    mean_reported = float(np.mean(reported))
    variation = (
        float(np.std(reported, ddof=1)) / mean_reported if mean_reported else 0.0
    )
    relative = math.sqrt(variation**2 / result.reps + 1.0 / (2.0 * (result.reps - 1)))
    return sigmas * relative


def assert_se_calibrated(
    result: MonteCarloResult,
    label: str = "",
    tolerance: float | None = None,
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if the reported standard error misstates the estimator's spread.

    Coverage can look correct while the reported standard error is wrong, if two
    errors cancel -- an inflated standard error paired with a bias, say. This
    checks the standard error directly against the spread actually observed.

    The tolerance comes from the replicate count by default, through the sampling
    distribution of the ratio: see :func:`se_ratio_tolerance`. Passing a number
    overrides it, which is worth doing only when the claim being tested is about
    a fixed accuracy -- "this sandwich estimator is within 5% at this sample
    size" -- rather than about the standard error being right.

    Args:
        result: A completed Monte Carlo study.
        label: Included in the failure message.
        tolerance: Largest permitted relative deviation of ``se_ratio`` from one.
            Derived from ``reps`` when omitted.
        sigmas: How many Monte Carlo standard errors of slack the derived
            tolerance allows. Ignored when ``tolerance`` is given.

    Raises:
        ValueError: If ``tolerance`` is not positive.
        AssertionError: If the ratio falls outside ``1 +- tolerance``, if the
            estimator did not vary at all across replicates, or if it reported no
            standard error to check.
    """
    if tolerance is not None and tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if not result.sampling_sd:
        raise AssertionError(
            f"{label}: the estimator did not vary across {result.reps} "
            "replicates, so its reported standard error cannot be checked "
            "against anything"
        )
    ratio = result.se_ratio
    if not np.isfinite(ratio):
        raise AssertionError(
            f"{label}: the estimator reported no usable standard error over "
            f"{result.reps} replicates (mean of the reported values is "
            f"{result.reported_se}), so there is nothing to check against its "
            f"observed spread of {result.sampling_sd:.6f}"
        )
    derived = tolerance is None
    band = se_ratio_tolerance(result, sigmas) if derived else float(tolerance)
    if abs(ratio - 1.0) > band:
        source = (
            f"{sigmas:g} Monte Carlo standard errors of the ratio at "
            f"{result.reps} replicates"
            if derived
            else "supplied by the caller"
        )
        raise AssertionError(
            f"{label}: reported standard error is {ratio:.3f} times the "
            f"observed spread ({result.reported_se:.6f} against "
            f"{result.sampling_sd:.6f}) over {result.reps} replicates; "
            f"tolerance is {band:.3f}, {source}"
        )


def assert_intervals_informative(
    result: MonteCarloResult,
    nominal: float = 0.95,
    label: str = "",
    max_ratio: float | None = None,
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if the intervals are so wide that their coverage means nothing.

    :func:`assert_coverage` is satisfied by an interval that always covers,
    whenever the study is small enough that a rate of 1.0 still sits inside the
    binomial band -- and it is *always* satisfied by ``coverage > 0.9`` written
    by hand. Three separate repositories worked around this with a comment
    saying so; one of them had shipped an inflation heuristic that drove the
    reported standard error to 3e7 times the estimation error while coverage
    stayed high, because a vacuous interval covers everything.

    Two things must both be true before this fails, and the conjunction is the
    point:

    1. **The interval is far wider than it needs to be.** ``width_ratio``, the
       mean width over the width a calibrated interval would have against this
       estimator's own spread, exceeds ``max_ratio`` by more than Monte Carlo
       noise. The default ``max_ratio`` is :func:`vacuous_width_ratio`, derived
       from ``nominal`` and ``reps``.
    2. **The study never once saw the interval fail.** Fewer than ``sigmas``
       misses in ``reps`` replicates: by the rule of three, a study observing no
       failures bounds the miss rate only at ``sigmas / reps``, so its coverage
       number is censored rather than measured.

    Requiring both is what keeps the gate off correct code. A Student t interval
    at ``n = 5`` is 1.33 times the normal oracle width, and an anytime-valid
    interval more, but both miss at their nominal rate, which the study sees, so
    neither is vacuous. Width alone cannot tell conservatism from vacuity;
    width plus a study that never saw a failure can.

    Args:
        result: A completed Monte Carlo study that recorded interval endpoints.
        nominal: The level the intervals claim.
        label: Included in the failure message.
        max_ratio: Override for the derived width multiple.
        sigmas: Monte Carlo slack on the width ratio, and the miss count below
            which the study is treated as never having seen a failure.

    Raises:
        ValueError: If the study recorded no interval endpoints, so there is no
            width to check, or if ``max_ratio`` is not positive.
        AssertionError: If the intervals are vacuous, or if the estimator did not
            vary at all across replicates, which leaves nothing to compare their
            width against.
    """
    lowers, uppers = result.lowers, result.uppers
    if lowers is None or uppers is None:
        raise ValueError(
            f"{label or 'this study'} recorded no interval endpoints, so the "
            "width of its intervals cannot be checked. Have the estimator "
            "report `lower` and `upper` on every replicate."
        )
    if max_ratio is not None and max_ratio <= 0:
        raise ValueError(f"max_ratio must be positive, got {max_ratio}")
    if not result.sampling_sd:
        raise AssertionError(
            f"{label}: the estimator did not vary across {result.reps} "
            "replicates, so there is no spread to compare its interval width "
            "against"
        )

    ratio = width_ratio(result, nominal)
    threshold = (
        vacuous_width_ratio(nominal, result.reps) if max_ratio is None else max_ratio
    )
    widths = result.widths
    mean_width = result.mean_width
    relative = math.sqrt(
        (float(np.var(widths, ddof=1)) / result.reps) / mean_width**2
        + 1.0 / (2.0 * (result.reps - 1))
        if mean_width
        else 0.0
    )
    if ratio - sigmas * ratio * relative <= threshold:
        return

    misses = int(np.count_nonzero((lowers > result.truth) | (result.truth > uppers)))
    if misses >= sigmas:
        return
    calibrated = 2.0 * _normal_tail_quantile((1.0 - nominal) / 2.0) * result.sampling_sd
    raise AssertionError(
        f"{label}: intervals are vacuous. Mean width {mean_width:.6g} is "
        f"{ratio:.3g} times the {calibrated:.6g} "
        f"a calibrated {nominal:.2f} interval needs against this estimator's "
        f"spread ({result.sampling_sd:.6g}), and the study saw {misses} misses "
        f"in {result.reps} replicates, so its coverage of "
        f"{1 - misses / result.reps:.3f} "
        f"measures the width rather than the estimator. The width at which a "
        f"study of this size stops being able to see a miss is "
        f"{threshold:.3g} times calibrated."
    )


def assert_narrower(
    narrow: MonteCarloResult,
    wide: MonteCarloResult,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail unless one method's intervals are measurably narrower than another's.

    The efficiency half of an interval comparison. Width without coverage is not
    a virtue -- the narrowest interval of all is the empty one -- so this is
    meant to be run *after* :func:`assert_coverage` on both studies, and it says
    nothing about either one's calibration.

    The tolerance is the Monte Carlo standard error of the difference in mean
    width over the two studies, so a gap this study cannot resolve does not pass,
    and the same call becomes stricter as the studies grow.

    Args:
        narrow: The study claimed to produce the narrower intervals.
        wide: The study it is claimed to beat.
        label: Included in the failure message.
        sigmas: How many Monte Carlo standard errors the gap must exceed.

    Raises:
        ValueError: If either study recorded no interval endpoints.
        AssertionError: If the narrower study's intervals are not measurably
            narrower.
    """
    for name, study in (("narrow", narrow), ("wide", wide)):
        if study.lowers is None or study.uppers is None:
            raise ValueError(
                f"{label or 'this comparison'}: the `{name}` study recorded no "
                "interval endpoints, so the two widths cannot be compared"
            )
    gap = wide.mean_width - narrow.mean_width
    standard_error = _mean_gap_se(wide.widths, narrow.widths)
    if not _gap_is_unresolved(gap, standard_error, sigmas):
        return
    resolved = (
        f"{gap / standard_error:+.2f} Monte Carlo standard errors"
        if standard_error
        else "an exactly zero Monte Carlo standard error"
    )
    raise AssertionError(
        f"{label}: mean width {narrow.mean_width:.6g} over {narrow.reps} "
        f"replicates is not measurably below {wide.mean_width:.6g} over "
        f"{wide.reps}: the gap of {gap:+.6g} is {resolved}, against a gate of "
        f"{sigmas:g}"
    )


def assert_power(
    result: MonteCarloResult,
    minimum: float,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if a test rejects less often than claimed under an alternative.

    One-sided, unlike :func:`assert_proportion`: power is a floor, not a target.
    Rejecting *more* often than the claim is not a defect of the test, and a
    two-sided band would fail an estimator for being better than promised. Size,
    which is a target, still belongs in :func:`assert_proportion`.

    The floor is the lower end of the binomial band around ``minimum``, so the
    claim is "power is at least ``minimum``, and this study can say so" rather
    than "the observed rate happened to clear ``minimum``". Under a claim that is
    exactly true the gate fires about once in 740 studies.

    Args:
        result: A completed Monte Carlo study run under the alternative.
        minimum: The power being claimed, in ``[0, 1]``.
        label: Included in the failure message.
        sigmas: Slack, in binomial standard errors.

    Raises:
        ValueError: If ``minimum`` is not in ``[0, 1]``, or if the study recorded
            no reject/accept decisions and so has no power to check.
        AssertionError: If the rejection rate falls below the floor.
    """
    if not 0.0 <= minimum <= 1.0:
        raise ValueError(f"minimum must be a probability, got {minimum}")
    if result.rejected is None:
        raise ValueError(
            f"{label or 'this study'} recorded no reject/accept decisions, so "
            "its power cannot be checked. Have the estimator report `rejected` "
            "on every replicate."
        )
    floor, _ = binomial_band(minimum, result.reps, sigmas)
    observed = result.rejection_rate
    if observed >= floor:
        return
    raise AssertionError(
        f"{label}: power {observed:.4f} is below the one-sided {sigmas:g}-sigma "
        f"floor {floor:.4f} for a claimed minimum of {minimum:.4f} over "
        f"{result.reps} replicates"
    )


def assert_more_powerful(
    more: MonteCarloResult,
    less: MonteCarloResult,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail unless one test rejects measurably more often than another.

    The comparison a method paper actually makes: at the same alternative, does A
    detect it more often than B. Asserting ``a.rejection_rate > b.rejection_rate``
    instead -- which is what this replaces -- passes on a gap of one replicate in
    four hundred, which is noise, and so certifies whichever method the seed
    happened to favour.

    Both studies must be run at the same alternative, which this cannot check.
    Comparing rejection rates under *different* alternatives compares the
    alternatives, not the tests.

    Args:
        more: The study claimed to be more powerful.
        less: The study it is claimed to beat.
        label: Included in the failure message.
        sigmas: How many standard errors of the difference the gap must exceed.

    Raises:
        ValueError: If either study recorded no reject/accept decisions.
        AssertionError: If the gap is not measurably positive.
    """
    for name, study in (("more", more), ("less", less)):
        if study.rejected is None:
            raise ValueError(
                f"{label or 'this comparison'}: the `{name}` study recorded no "
                "reject/accept decisions, so the two cannot be compared"
            )
    strong, weak = more.rejection_rate, less.rejection_rate
    gap = strong - weak
    standard_error = math.sqrt(
        strong * (1.0 - strong) / more.reps + weak * (1.0 - weak) / less.reps
    )
    if not _gap_is_unresolved(gap, standard_error, sigmas):
        return
    resolved = (
        f"{gap / standard_error:+.2f} standard errors of the difference"
        if standard_error
        else "an exactly zero standard error"
    )
    raise AssertionError(
        f"{label}: rejection rate {strong:.4f} over {more.reps} replicates is "
        f"not measurably above {weak:.4f} over {less.reps}: the gap of "
        f"{gap:+.4f} is {resolved}, against a gate of {sigmas:g}"
    )
