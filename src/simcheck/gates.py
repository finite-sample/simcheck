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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .results import MonteCarloResult

__all__ = [
    "GATE_SIGMAS",
    "assert_count_rate",
    "assert_coverage",
    "assert_proportion",
    "assert_se_calibrated",
    "assert_unbiased",
    "binomial_band",
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
    low, high = binomial_band(nominal, reps, sigmas)
    if not low <= observed <= high:
        raise AssertionError(
            f"{label}: observed rate {observed:.4f} outside the {sigmas:g}-sigma "
            f"band [{low:.4f}, {high:.4f}] for a nominal {nominal:.4f} "
            f"over {reps} replicates"
        )


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
    assert_proportion(successes / reps, reps, nominal, label, sigmas)


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
    """
    assert_proportion(
        result.coverage, result.reps, nominal, f"{label} coverage".strip(), sigmas
    )


def assert_se_calibrated(
    result: MonteCarloResult,
    label: str = "",
    tolerance: float = 0.15,
) -> None:
    """Fail if the reported standard error misstates the estimator's spread.

    Coverage can look correct while the reported standard error is wrong, if two
    errors cancel -- an inflated standard error paired with a bias, say. This
    checks the standard error directly against the spread actually observed.

    Args:
        result: A completed Monte Carlo study.
        label: Included in the failure message.
        tolerance: Largest permitted relative deviation of ``se_ratio`` from one.

    Raises:
        ValueError: If ``tolerance`` is not positive.
        AssertionError: If the ratio falls outside ``1 +- tolerance``, or the
            estimator did not vary at all across replicates.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    ratio = result.se_ratio
    if not np.isfinite(ratio):
        raise AssertionError(
            f"{label}: the estimator did not vary across {result.reps} "
            "replicates, so its reported standard error cannot be checked "
            "against anything"
        )
    if abs(ratio - 1.0) > tolerance:
        raise AssertionError(
            f"{label}: reported standard error is {ratio:.3f} times the "
            f"observed spread ({result.reported_se:.6f} against "
            f"{result.sampling_sd:.6f}) over {result.reps} replicates; "
            f"tolerance is {tolerance:.2f}"
        )
