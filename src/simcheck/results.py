"""What a Monte Carlo study of an estimator produced.

One :class:`MonteCarloResult` holds the per-replicate output of a study at a
single point, and derives from it the quantities the assertions in
:mod:`simcheck.gates` test: bias, the ratio of claimed to actual standard error,
coverage, and rejection rate.

**One point per study, deliberately.** It is tempting to pool every point of a
fitted curve into one coverage number. Whether an interval covers at ``x=40`` and
whether it covers at ``x=41`` are very nearly the same event, though -- they come
from one curve fitted to one dataset -- so pooling multiplies the apparent sample
size while adding almost no information, and makes a badly calibrated estimator
look precisely measured. Keep one binomial per point and the arithmetic stays
honest.

**The endpoints are kept, not just the hit.** Coverage on its own cannot tell a
calibrated interval from one so wide it could not have failed, and reducing each
replicate to a boolean throws away the only evidence that would. Keeping
``lowers`` and ``uppers`` is what lets :func:`~simcheck.assert_intervals_informative`
measure the width against the spread the estimator actually has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["MonteCarloResult"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Sampling behaviour of an estimator at one point.

    Args:
        estimates: The estimate from each replicate.
        standard_errors: The standard error the estimator reported on each replicate.
        covered: Whether each replicate's interval contained the truth, or None
            when the estimator reported no intervals. None rather than an array
            of False: the latter reads as coverage of zero, which is a broken
            estimator, not an absent measurement.
        rejected: Whether each replicate rejected the null, or None when the
            estimator performs no test.
        truth: The true value of the quantity being estimated.
        lowers: Lower endpoint of each replicate's interval, or None when the
            estimator reported no intervals. Supplying the endpoints is what
            makes the interval's *width* checkable, and coverage alone cannot
            distinguish a calibrated interval from a vacuous one.
        uppers: Upper endpoint of each replicate's interval. Both endpoints or
            neither; one without the other is not an interval.

    Raises:
        ValueError: If the arrays disagree in length or are empty, if one
            endpoint array is given without the other, if any interval runs
            backwards, or if ``covered`` contradicts the endpoints.
    """

    estimates: npt.NDArray[np.float64]
    standard_errors: npt.NDArray[np.float64]
    covered: npt.NDArray[np.bool_] | None
    rejected: npt.NDArray[np.bool_] | None
    truth: float
    lowers: npt.NDArray[np.float64] | None = None
    uppers: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Reject ragged, empty or self-contradictory input.

        When the endpoints are given and ``covered`` is not, ``covered`` is
        filled in from them. That is not a guess: the endpoints and the truth
        determine coverage exactly, and leaving the field None would make a
        study that plainly measured coverage report that it had not.

        Raises:
            ValueError: If the arrays disagree in length or are empty, if one
                endpoint array is given without the other, if any interval runs
                backwards, or if ``covered`` contradicts the endpoints.
        """
        if (self.lowers is None) != (self.uppers is None):
            raise ValueError(
                "an interval needs both endpoints: got "
                f"lowers={'None' if self.lowers is None else 'an array'} and "
                f"uppers={'None' if self.uppers is None else 'an array'}"
            )
        lengths = {
            len(self.estimates),
            len(self.standard_errors),
            *([] if self.covered is None else [len(self.covered)]),
            *([] if self.rejected is None else [len(self.rejected)]),
            *([] if self.lowers is None else [len(self.lowers)]),
            *([] if self.uppers is None else [len(self.uppers)]),
        }
        if len(lengths) > 1:
            raise ValueError(
                f"per-replicate arrays disagree in length: {sorted(lengths)}"
            )
        if lengths == {0}:
            raise ValueError("a Monte Carlo study needs at least one replicate")
        if self.lowers is None or self.uppers is None:
            return

        backwards = int(np.count_nonzero(self.uppers < self.lowers))
        if backwards:
            raise ValueError(
                f"{backwards} of {len(self.lowers)} intervals have an upper "
                "endpoint below the lower one, so their width is negative"
            )
        hit = (self.lowers <= self.truth) & (self.truth <= self.uppers)
        if self.covered is None:
            object.__setattr__(self, "covered", hit)
            return
        disagreements = int(np.count_nonzero(np.asarray(self.covered) != hit))
        if disagreements:
            raise ValueError(
                f"`covered` disagrees with the endpoints on {disagreements} of "
                f"{len(hit)} replicates. One of the two is measuring something "
                "the other is not -- an interval on a transformed scale, say -- "
                "and pooling them would report a coverage rate for one and a "
                "width for the other."
            )

    @property
    def reps(self) -> int:
        """Number of replicates."""
        return len(self.estimates)

    @property
    def bias(self) -> float:
        """Mean deviation of the estimate from the truth."""
        return float(np.mean(self.estimates) - self.truth)

    @property
    def sampling_sd(self) -> float:
        """The estimator's actual spread across replicates."""
        if self.reps < 2:
            return 0.0
        return float(np.std(self.estimates, ddof=1))

    @property
    def reported_se(self) -> float:
        """The standard error the estimator claims, on average."""
        return float(np.mean(self.standard_errors))

    @property
    def se_ratio(self) -> float:
        """Claimed standard error over actual spread. One means calibrated.

        Below one, the estimator is overconfident and its intervals will
        under-cover; above one it is conservative.
        """
        return self.reported_se / self.sampling_sd if self.sampling_sd else np.nan

    @property
    def mc_se(self) -> float:
        """Monte Carlo standard error of the mean estimate.

        This is the precision of the *study*, not of the estimator: it is what
        makes ``bias`` interpretable, because a bias smaller than this is
        indistinguishable from simulation noise.
        """
        if not self.sampling_sd:
            return 0.0
        return self.sampling_sd / float(np.sqrt(self.reps))

    @property
    def bias_t(self) -> float:
        """Bias in units of its own Monte Carlo standard error.

        A t statistic for the null that the estimator is unbiased. Zero when the
        estimator is deterministic, in which case there is no sampling variation
        to test against.
        """
        return self.bias / self.mc_se if self.mc_se else 0.0

    @property
    def coverage(self) -> float:
        """Fraction of replicates whose interval contained the truth.

        Returns:
            float: The coverage rate.

        Raises:
            ValueError: If the study recorded no intervals. Reporting zero here
                would be indistinguishable from an estimator whose intervals
                never cover, which is the opposite conclusion.
        """
        if self.covered is None:
            raise ValueError(
                "this study recorded no confidence intervals, so it has no "
                "coverage rate. Have the estimator report `lower` and `upper` "
                "on every replicate."
            )
        return float(np.mean(self.covered))

    @property
    def widths(self) -> npt.NDArray[np.float64]:
        """Width of each replicate's interval.

        Returns:
            numpy.ndarray: One width per replicate.

        Raises:
            ValueError: If the study did not record interval endpoints. Reporting
                zeros here would read as an infinitely precise estimator, which
                is the opposite of an unmeasured one.
        """
        if self.lowers is None or self.uppers is None:
            raise ValueError(
                "this study recorded no interval endpoints, so it has no "
                "widths. Have the estimator report `lower` and `upper` on every "
                "replicate."
            )
        return np.asarray(self.uppers - self.lowers, dtype=float)

    @property
    def mean_width(self) -> float:
        """Mean interval width, in the units of the estimand.

        Reading this on a study that recorded no endpoints raises ``ValueError``
        through :attr:`widths`, rather than reporting a width of zero.

        Returns:
            float: The mean width.
        """
        return float(np.mean(self.widths))

    @property
    def median_width(self) -> float:
        """Median interval width.

        Worth having alongside the mean: a procedure that returns an enormous
        interval on a few replicates has a mean width dominated by those, and the
        median says what the typical replicate produced. Reading it on a study
        that recorded no endpoints raises ``ValueError`` through :attr:`widths`.

        Returns:
            float: The median width.
        """
        return float(np.median(self.widths))

    @property
    def rejection_rate(self) -> float:
        """Fraction of replicates that rejected the null.

        Under a true null this estimates the test's size; under an alternative,
        its power.

        Returns:
            float: The rejection rate.

        Raises:
            ValueError: If the study recorded no reject/accept decisions.
        """
        if self.rejected is None:
            raise ValueError(
                "this study recorded no reject/accept decisions, so it has no "
                "rejection rate."
            )
        return float(np.mean(self.rejected))

    def summary(self, label: str = "") -> str:
        """A one-line report, for printing a table of results.

        Args:
            label: Row label.

        Returns:
            str: One formatted line.
        """
        cover = "  n/a" if self.covered is None else f"{self.coverage:5.3f}"
        reject = "  n/a" if self.rejected is None else f"{self.rejection_rate:5.3f}"
        return (
            f"{label:34s} bias={self.bias:+9.5f} t={self.bias_t:+6.2f} "
            f"SE/MC={self.se_ratio:5.3f} cover={cover} reject={reject}"
        )
