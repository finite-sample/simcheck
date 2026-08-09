# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-08

### Added

- Initial extraction from `incline/tests/_statistics.py`, generalised so it
  depends on numpy alone rather than on any one estimator's interfaces.

- `MonteCarloResult`, holding one study's per-replicate output and deriving bias,
  Monte Carlo standard error, the claimed-to-actual standard error ratio,
  coverage and rejection rate from it. It rejects ragged and empty input, which
  the original silently accepted.

- Gates whose tolerance comes from the replicate count rather than from a
  hand-picked threshold: `assert_unbiased`, `assert_coverage`,
  `assert_se_calibrated`, `assert_proportion`, `assert_count_rate` and the
  `binomial_band` they are built on.

- `assert_se_calibrated`, which has no counterpart in the original. Coverage can
  look correct while the reported standard error is wrong, when an inflated
  standard error is cancelled by a bias; this checks the standard error against
  the spread actually observed.

- Tier helpers `reps_for`, `deep_tier`, `FAST_REPS` and `DEEP_REPS`, driven by
  `SIMCHECK_DEEP` and `SIMCHECK_REPS`.

- **Interval endpoints are kept, not just whether each interval covered.**
  `MonteCarloResult` gained `lowers`, `uppers`, `widths`, `mean_width` and
  `median_width`; `monte_carlo` passes the endpoints through instead of computing
  them, using them once and dropping them. Supplying one endpoint without the
  other, an upper endpoint below its lower one, or a `covered` array that
  contradicts the endpoints are all rejected; supplying endpoints without
  `covered` fills `covered` in, since the endpoints and the truth determine it.

- `assert_intervals_informative`, which fails an interval so wide that the study
  never saw it miss. `assert_coverage` cannot: a coverage rate of 1.0 sits inside
  the binomial band at any study smaller than about 60 replicates, and a
  hand-written `coverage > 0.9` is satisfied by an interval that always covers.
  Three consuming repositories hit this independently and left a comment where a
  test should have been — one of them after an inflation heuristic drove a
  reported standard error to 3e7 times the estimation error while coverage stayed
  high. The gate fails only when the width exceeds `vacuous_width_ratio(nominal,
  reps)` *and* the study observed fewer than three misses, so a correct but
  conservative procedure — a t interval at n=5 is 1.33 times the oracle width —
  is not flagged.

- `vacuous_width_ratio` and `width_ratio`, the threshold and the measured
  quantity behind that gate. The reference width, `2 * z * sampling_sd`, is
  measured by the study, so no absolute width appears anywhere. How many expected
  misses per study counts as "could not have failed" is a convention rather than
  a derivation, and `vacuous_width_ratio`'s docstring says so and says why: correct
  procedures occupy the whole range of widths above the oracle, so no sampling
  distribution separates conservatism from vacuity on width alone.

- `assert_narrower`, for the efficiency half of an interval comparison, banded by
  the Monte Carlo standard error of the difference in mean width.

- `assert_power` and `assert_more_powerful`. The package documented power as one
  of the four questions it answers and had no gate for it; consumers were
  reaching for `assert_proportion`, which is two-sided and needs a nominal you
  already know analytically, or hand-rolling a two-sample standard error.
  `assert_power` is one-sided, because power is a floor and a two-sided band
  fails a test for being better than claimed.

- **Negative tests for every gate.** Each is exercised on input that satisfies
  its property, where it must stay silent, and on input that violates it, where
  it must raise. Plus gates run against estimators whose behaviour is known
  analytically: the sample mean trips nothing, the `ddof=0` variance is caught
  with its textbook `-sigma^2/n` bias, a 1.96 interval at n=5 is caught
  under-covering at about 0.875, a two-sided z test shows the power its formula
  gives (0.323 at n=25 and 0.851 at n=100 for delta=0.3), an interval built at a
  known scale comes out at a width ratio of exactly one, and a false-positive
  check confirms correct estimators are essentially never flagged.

### Changed

- **`assert_se_calibrated`'s tolerance now comes from the replicate count.** It
  was `0.15`, the one number in the package chosen by hand rather than derived,
  and it was wrong in both directions: `se_ratio` divides a mean of `reps`
  reported standard errors by a sample standard deviation of `reps` estimates, so
  its Monte Carlo spread is `sqrt(cv^2/reps + (kappa-1)/(4*reps))`, and three of
  those is 0.21 at 100 replicates and 0.05 at 2000 for a normal estimator. The fixed value was therefore tight
  enough to fail correct estimators in a fast tier and loose enough to certify a
  12% error in a deep one. Passing `tolerance=` explicitly still overrides it,
  and `se_ratio_tolerance(result)` returns the derived band.

  This changes behaviour for callers that relied on the default. Of the six
  consuming repositories only `geoinference` does, and its suite was re-run
  against this branch: 43 passed, 4 subtests passed. Every other consumer passes
  `tolerance=` explicitly or does not call the gate.

  The band uses the estimator's own fourth moment, not a normal assumption:
  `Var(s)/sigma^2 = (kappa-1)/(4*reps)`, with `kappa` estimated from the study and
  floored at 3 so a downward-biased sample kurtosis cannot narrow it. Assuming
  normality flagged a *calibrated* estimator with Student t(5) sampling error in
  19 of 200 studies; with the fourth-moment term it is 2 of 200.

### Fixed

- **`assert_se_calibrated` diagnosed a missing standard error as a constant
  estimator.** `Estimate` documents that leaving `standard_error` as NaN means
  the gate "will have nothing to check and will say so"; it said the estimator
  did not vary across replicates, which is a different and false diagnosis.

- **The extracted `assert_rate` could report the worst possible result as the
  best possible one.** It took either a count or a rate and guessed which:
  `observed = successes / reps if successes > 1 else float(successes)`. For a
  nominal 95% study, one hit in four hundred replicates — a total failure of the
  estimator — is not greater than one, so it was read as a rate of 1.0, which
  sits inside the band, and the assertion passed.

  No heuristic can separate a count of 1 from a rate of 1.0, so counts and rates
  are now separate functions, and `assert_proportion` raises rather than guesses
  when handed a value outside `[0, 1]`.
