"""Run an estimator over many synthetic datasets and record what it did.

The loop itself is trivial; the two decisions in it are not.

**Each replicate gets its own independent generator**, spawned from a seed
sequence rather than drawn from one shared stream. With a shared stream,
replicate 400 depends on every draw before it, so you cannot reproduce it alone,
and changing the replicate count changes every replicate rather than adding to
them. Spawning makes replicate *i* a function of ``(seed, i)`` and nothing else.

**A missing interval is an error, not a miss.** If an estimator reports no
confidence interval, the obvious implementation records ``covered = False`` for
that replicate. Coverage then comes out at zero and the assertion fails with
"coverage 0.000 outside the band" -- which reads as a catastrophically broken
estimator rather than as a study that never recorded intervals at all. Here a
study either records intervals for every replicate or for none, and asking a
study with none for its coverage raises and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .results import MonteCarloResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

__all__ = ["Estimate", "monte_carlo"]


@dataclass(frozen=True)
class Estimate:
    """What one replicate of an estimator produced.

    Args:
        value: The point estimate.
        standard_error: The standard error the estimator reported. Leave as NaN
            when it reports none; :func:`~simcheck.assert_se_calibrated` will
            then have nothing to check and will say so.
        lower: Lower end of the interval, if the estimator produced one.
        upper: Upper end of the interval.
        rejected: Whether the replicate rejected the null, if the estimator
            performs a test.
    """

    value: float
    standard_error: float = float("nan")
    lower: float | None = None
    upper: float | None = None
    rejected: bool | None = None

    @property
    def has_interval(self) -> bool:
        """Whether this replicate reported both interval endpoints."""
        return self.lower is not None and self.upper is not None


def monte_carlo(
    replicate: Callable[[np.random.Generator], Estimate],
    truth: float,
    reps: int,
    *,
    seed: int = 0,
) -> MonteCarloResult:
    """Run ``replicate`` many times and collect its sampling behaviour.

    Args:
        replicate: Called once per replicate with its own generator. It should
            simulate a dataset, fit the estimator, and return what came out. It
            must not close over shared mutable state, or the replicates stop
            being independent.
        truth: The true value of the quantity being estimated, which the caller
            knows because it generated the data.
        reps: Number of replicates.
        seed: Seed for the replicate stream. Replicate ``i`` is a function of
            ``(seed, i)`` alone, so it can be reproduced on its own and raising
            ``reps`` adds replicates rather than changing the existing ones.

    Returns:
        MonteCarloResult: The recorded sampling behaviour.

    Raises:
        ValueError: If ``reps`` is not positive, or if the estimator reported an
            interval on some replicates but not others -- which means it is not
            doing the same thing every time, and pooling the results would
            silently mix two estimators.

    Examples:
        >>> import numpy as np
        >>> from simcheck import Estimate, assert_unbiased, monte_carlo
        >>> def draw(rng):
        ...     x = rng.normal(2.0, 1.0, 50)
        ...     se = x.std(ddof=1) / np.sqrt(50)
        ...     return Estimate(x.mean(), se, x.mean() - 2 * se, x.mean() + 2 * se)
        >>> result = monte_carlo(draw, truth=2.0, reps=200, seed=0)
        >>> result.reps
        200
        >>> assert_unbiased(result, "sample mean")
    """
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")

    children = np.random.SeedSequence(seed).spawn(reps)

    values = np.empty(reps, dtype=float)
    errors = np.empty(reps, dtype=float)
    lowers = np.empty(reps, dtype=float)
    uppers = np.empty(reps, dtype=float)
    flags = np.zeros(reps, dtype=bool)
    with_interval = np.zeros(reps, dtype=bool)
    with_decision = np.zeros(reps, dtype=bool)

    for i, child in enumerate(children):
        outcome = replicate(np.random.default_rng(child))
        values[i] = outcome.value
        errors[i] = outcome.standard_error
        with_interval[i] = outcome.has_interval
        if outcome.has_interval:
            lowers[i] = outcome.lower  # type: ignore[assignment]
            uppers[i] = outcome.upper  # type: ignore[assignment]
        if outcome.rejected is not None:
            with_decision[i] = True
            flags[i] = bool(outcome.rejected)

    covered = None
    if with_interval.all():
        covered = (lowers <= truth) & (truth <= uppers)
    elif with_interval.any():
        raise ValueError(
            f"{int(with_interval.sum())} of {reps} replicates reported an "
            "interval and the rest did not. Coverage over a mixture of the two "
            "is not a coverage rate; make the estimator report an interval "
            "every time, or none of the time."
        )

    rejected = None
    if with_decision.all():
        rejected = flags
    elif with_decision.any():
        raise ValueError(
            f"{int(with_decision.sum())} of {reps} replicates reported a "
            "reject/accept decision and the rest did not. A rejection rate over "
            "a mixture is not a size or a power."
        )

    return MonteCarloResult(values, errors, covered, rejected, truth)
