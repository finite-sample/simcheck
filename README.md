# simcheck

**Monte Carlo tests for statistical estimators.**

The tests most statistical code ships assert that it runs. The tests it needs
assert the classical properties: if the assumptions hold, is the estimator
unbiased, do its intervals cover at the nominal rate, is the test's size right
under the null, and does it have power under an alternative.

```bash
pip install simcheck
# until the first PyPI release lands:
pip install "simcheck @ git+https://github.com/finite-sample/simcheck@v0.1.0"
```

Python 3.11+. Depends on numpy, and nothing else.

## The rule

**The tolerance comes from the replicate count, never from a number you picked.**

```python
assert coverage > 0.85  # for a nominal 0.95. What does this test?
```

Nothing useful. At ten thousand replicates it passes an estimator that covers
86% of the time — badly broken. At fifty replicates it fails a perfectly correct
one about a fifth of the time. Every gate in simcheck derives its band from the
number of replicates instead, so the same line adapts to the tier it runs in:

```python
from simcheck import MonteCarloResult, assert_coverage, assert_unbiased

result = MonteCarloResult(
    estimates=means,  # one per replicate
    standard_errors=errors,  # what the estimator claimed, per replicate
    covered=hits,  # did the interval contain the truth
    rejected=flags,  # did the test reject
    truth=2.0,
)

assert_unbiased(result, "sample mean")
assert_coverage(result, 0.95, "t interval")
```

A failure says how far outside the band it fell, in standard errors, so you can
tell "broken" from "slightly miscalibrated" without rerunning anything.

## Running the study

`monte_carlo` does the loop, and makes two decisions worth knowing about:

```python
import numpy as np
from simcheck import Estimate, monte_carlo, assert_coverage, assert_unbiased


def replicate(rng):
    x = rng.normal(2.0, 1.0, 40)
    se = x.std(ddof=1) / np.sqrt(40)
    return Estimate(x.mean(), se, x.mean() - 1.96 * se, x.mean() + 1.96 * se)


result = monte_carlo(replicate, truth=2.0, reps=2000, seed=11)
assert_unbiased(result, "sample mean")
assert_coverage(result, 0.95, "1.96 interval at n=40")
```

**Replicate *i* depends on `(seed, i)` and nothing else.** Each gets its own
generator, spawned from a seed sequence rather than drawn from one shared
stream. With a shared stream, replicate 400 depends on every draw before it: you
cannot reproduce it alone, and raising the replicate count changes every
existing replicate instead of adding to them. Here `reps=500` extends `reps=50`
— the first fifty are identical, and a test asserts it.

**A missing interval is an error, not a miss.** If your estimator reports no
interval, the obvious implementation records `covered = False`, coverage comes
out at 0.000, and the assertion fails with "outside the band" — which reads as a
catastrophically broken estimator rather than a study that never measured
coverage. `monte_carlo` records `covered = None` instead, and asking for the
coverage rate raises and says why. An estimator that reports an interval only
*sometimes* is rejected outright: averaging over that mixture is not a coverage
rate.

## What it checks

| Gate | Fails when |
|---|---|
| `assert_unbiased` | the mean estimate is more than 3 Monte Carlo standard errors from the truth |
| `assert_coverage` | interval coverage falls outside the binomial band around the nominal level |
| `assert_intervals_informative` | the intervals are so wide the study never saw one miss, so their coverage measures the width |
| `assert_narrower` | one method's intervals are not measurably narrower than another's |
| `assert_se_calibrated` | the reported standard error misstates the spread actually observed |
| `assert_power` | a test rejects less often than claimed under an alternative (one-sided) |
| `assert_more_powerful` | one test does not reject measurably more often than another at the same alternative |
| `assert_proportion` | an observed **rate** — size, coverage — is inconsistent with the claimed one |
| `assert_count_rate` | the same, given a **count** of successes |

`binomial_band(nominal, reps)`, `vacuous_width_ratio(nominal, reps)` and
`se_ratio_tolerance(result)` give the three thresholds directly, if you want to
assert something else against them. `width_ratio(result, nominal)` is the
measured quantity behind the first of those.

## Coverage cannot see a vacuous interval

`assert_coverage` is satisfied by an interval so wide it always covers, whenever
the study is small enough that a rate of 1.0 still sits inside the binomial band
— and `assert coverage > 0.9`, written by hand, is satisfied by it always.
Three repositories hit this independently and each left a comment where a test
should have been; one had shipped an inflation heuristic that drove the reported
standard error to 3×10⁷ times the estimation error while coverage stayed high,
because a vacuous interval covers everything.

The fix is to keep the endpoints, not just the hit:

```python
result = monte_carlo(replicate, truth=2.0, reps=2000, seed=11)
assert_coverage(result, 0.95, "t interval")
assert_intervals_informative(result, 0.95, "t interval")
```

Two things must both be true before that fails, and the conjunction is the
point. The interval must be far wider than the width its own level requires
against the spread the estimator actually has — `2 * z * sampling_sd`, measured
by the study, so no absolute width is written down anywhere. *And* the study must
never have seen it miss. A Student t interval at n=5 is 1.33 times the normal
oracle width and an anytime-valid interval is wider still; both are correct, both
miss at their nominal rate, and the study watches them do it. Width alone cannot
tell conservatism from vacuity. Width plus a study that never saw a failure can.

`vacuous_width_ratio(0.95, reps)` is the width multiple at which a study of that
size stops being able to observe a miss: 1.78 at 100 replicates, 1.96 at 400,
2.15 at 2000. It rises with the replicate count, which inverts the usual
direction and is meant to — more replicates resolve rarer failures.

The one thing here that is a judgement rather than a derivation is how many
expected misses per study counts as "could not have failed". Its docstring says
so, and says why no sampling distribution fixes it: correct procedures occupy the
whole range of widths above the oracle.

## Power

The package claims to answer whether a test has power under an alternative, so
there is a gate for it:

```python
assert_power(result, 0.80, "score test at delta=0.3")
assert_more_powerful(robust, naive, "robust against naive at the same alternative")
```

`assert_power` is one-sided, unlike `assert_proportion`: power is a floor, and a
two-sided band would fail a test for being *better* than promised. Size, which is
a target rather than a floor, still belongs in `assert_proportion`.

`assert_more_powerful` bands the gap between two rejection rates by the standard
error of the difference. The assertion it replaces — `a.rejection_rate >
b.rejection_rate` — passes on a gap of one replicate in four hundred and reports
whichever method the seed favoured as the winner.

## Two failure modes this is built to prevent

Both were found in production code, and both are why the package exists.

**An aggregate threshold absorbing a systematic failure.** A test matrix
asserting `success_rate >= 0.7` passed for several releases while one input
pattern in eight raised an exception for every single configuration — 216 of
1296 combinations had never run. One eighth is 12.5%, comfortably inside a 30%
allowance, so every configuration scored exactly 87.5% and the suite reported
success. Assert the property, not a rate with room to hide things in.

**An assertion helper that cannot fail.** The helper simcheck was extracted from
accepted either a count or a rate and guessed which:

```python
observed = successes / reps if successes > 1 else float(successes)
```

One hit in four hundred replicates against a nominal 95% is a total failure of
the estimator. One is not greater than one, so it was read as a *rate* of 1.0 —
inside the band — and the assertion passed. The worst possible result was
reported as the best possible one. That is why counts and rates are separate
functions here, and why `assert_proportion` raises rather than guesses when it
is handed something outside `[0, 1]`.

## The rule applies to simcheck too

`assert_se_calibrated` used to take `tolerance=0.15`, which was the one number in
the package chosen by hand rather than derived — and it was wrong in both
directions at once. `se_ratio` is `mean(reported se) / sd(estimates)`, and both
halves are estimated from the same replicates, so it is noisy even when the
estimator is perfect: the numerator's relative standard error is `cv/sqrt(reps)`
and the denominator's is `sqrt((κ-1)/(4·reps))`, where κ is the estimator's
kurtosis — measured, not assumed, because a normal assumption flags a correct
heavy-tailed estimator. Added in quadrature and taken at three sigma, that is
0.21 at 100 replicates and 0.05 at 2000 for a normal estimator. A fixed 0.15 was
therefore tight enough to fail correct estimators in a fast tier and loose enough
to certify a 12% error in a deep one.

The tolerance is now derived from `reps` by default; passing a number still
overrides it, which is worth doing when the claim really is about a fixed
accuracy at a fixed sample size. `se_ratio_tolerance(result)` returns the band.

## Negative tests

Every gate has one: an input that violates the property, and a check that the
gate raises on it. `tests/test_negative.py` is the file that makes the rest of
the package worth anything — a helper that silently passes everything is worse
than no helper, because it converts an untested codebase into one that reports
itself as tested.

The gates are also run against estimators whose behaviour is known analytically
(`tests/test_against_known_statistics.py`): the sample mean must trip nothing,
the uncorrected `ddof=0` variance must be caught with its textbook bias of
`-σ²/n`, a 1.96 interval at n=5 must be caught under-covering at ≈0.875, a
two-sided z test must show the power its formula gives (0.323 at n=25, 0.851 at
n=100 for δ=0.3), and an interval built at a *known* scale must come out at a
width ratio of exactly one. There is also a false-positive check, because a gate
that fires on 5% of correct code gets disabled within a week.

## Tiers

```python
from simcheck import reps_for

reps = reps_for()  # 100 normally; 400 when SIMCHECK_DEEP is set
```

`SIMCHECK_REPS` overrides the deep count. Because the gates derive tolerance
from the replicate count, raising it in a scheduled job makes every assertion
stricter without touching a line of test code — and a test cannot be quietly
weakened by lowering it, since the band widens visibly in the failure message.

## License

MIT.
