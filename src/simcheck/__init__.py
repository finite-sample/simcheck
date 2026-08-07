"""Monte Carlo tests for statistical estimators.

The tests most statistical code ships assert that it runs. The tests it needs
assert the classical properties: if the assumptions hold, is the estimator
unbiased, do its intervals cover at the nominal rate, is the test's size right
under the null, and does it have power under an alternative.

simcheck supplies the machinery for those four questions, with one rule running
through it: **the tolerance comes from the replicate count**, never from a number
chosen by hand. ``coverage > 0.85`` for a nominal 95% interval is meaningless --
far too loose at ten thousand replicates, tight enough to fail spuriously at
fifty. Every gate here derives its band from the number of replicates, so the
same assertion adapts to the tier it runs in and reports how far outside the band
it fell.

Two failure modes this is built to prevent, both observed in the wild:

**An aggregate threshold absorbing a systematic failure.** A matrix test
asserting ``success_rate >= 0.7`` passed for releases while one input pattern in
eight raised an exception for every configuration -- 12.5% sits comfortably
inside a 30% allowance. Assert the property, not a rate that has room to hide
things.

**An assertion helper that cannot fail.** Every gate here has a negative test: an
input that violates the property, and a check that the gate raises on it. A
helper that silently passes everything is worse than no helper, because it
converts an untested codebase into one that reports itself as tested.

Examples:
>>> import numpy as np
>>> from simcheck import MonteCarloResult, assert_coverage, assert_unbiased
>>> rng = np.random.default_rng(0)
>>> truth, n, reps = 2.0, 40, 400
>>> draws = rng.normal(truth, 1.0, size=(reps, n))
>>> means = draws.mean(axis=1)
>>> errors = draws.std(axis=1, ddof=1) / np.sqrt(n)
>>> result = MonteCarloResult(
...     estimates=means,
...     standard_errors=errors,
...     covered=np.abs(means - truth) <= 1.96 * errors,
...     rejected=np.abs(means) > 1.96 * errors,
...     truth=truth,
... )
>>> assert_unbiased(result, "sample mean")
>>> assert_coverage(result, 0.95, "sample mean")
"""

from importlib.metadata import PackageNotFoundError, version

from .gates import (
    GATE_SIGMAS,
    assert_count_rate,
    assert_coverage,
    assert_proportion,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
)
from .results import MonteCarloResult
from .runner import Estimate, monte_carlo
from .tiers import DEEP_REPS, FAST_REPS, deep_tier, reps_for

try:
    __version__ = version("simcheck")
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0"

__all__ = [
    "DEEP_REPS",
    "FAST_REPS",
    "GATE_SIGMAS",
    "Estimate",
    "MonteCarloResult",
    "__version__",
    "assert_count_rate",
    "assert_coverage",
    "assert_proportion",
    "assert_se_calibrated",
    "assert_unbiased",
    "binomial_band",
    "deep_tier",
    "monte_carlo",
    "reps_for",
]
