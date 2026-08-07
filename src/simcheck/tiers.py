"""How many replicates to run, and where that number comes from.

Monte Carlo tests are slow, so they get run at two sizes: a fast tier that runs
on every commit and a deep tier that runs on a schedule or before a release.

The replicate count is a *variable*, not a constant baked into each test, because
the gates in :mod:`simcheck.gates` derive their tolerance from it. Raising
``SIMCHECK_REPS`` in a scheduled job therefore makes every assertion stricter
without editing a line of test code -- and, importantly, a test written this way
cannot be quietly weakened by lowering the replicate count, because the band
widens visibly in the failure message.
"""

from __future__ import annotations

import os

__all__ = ["DEEP_REPS", "FAST_REPS", "deep_tier", "reps_for"]

# Enough to catch a grossly wrong estimator in seconds, and no more. At 100
# replicates a 3-sigma band around a nominal 0.95 spans [0.88, 1.00], so this
# tier detects a broken estimator, not a slightly miscalibrated one.
FAST_REPS = 100

# The tier that can actually resolve a small miscalibration. Overridable so CI
# can run a deeper study than a laptop without the tests changing.
DEEP_REPS = int(os.environ.get("SIMCHECK_REPS", "400"))


def deep_tier() -> bool:
    """Whether the deep tier has been requested.

    Returns:
        bool: True when ``SIMCHECK_DEEP`` is set to something truthy.
    """
    return os.environ.get("SIMCHECK_DEEP", "").lower() in {"1", "true", "yes", "on"}


def reps_for(deep: bool | None = None) -> int:
    """Replicate count for the current tier.

    Args:
        deep: Force a tier. When omitted, read the environment.

    Returns:
        int: Number of replicates to run.
    """
    use_deep = deep_tier() if deep is None else deep
    return DEEP_REPS if use_deep else FAST_REPS
