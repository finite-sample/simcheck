# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **Negative tests for every gate.** Each is exercised on input that satisfies
  its property, where it must stay silent, and on input that violates it, where
  it must raise. Plus gates run against estimators whose behaviour is known
  analytically: the sample mean trips nothing, the `ddof=0` variance is caught
  with its textbook `-sigma^2/n` bias, a 1.96 interval at n=5 is caught
  under-covering at about 0.875, and a false-positive check confirms correct
  estimators are essentially never flagged.

### Fixed

- **The extracted `assert_rate` could report the worst possible result as the
  best possible one.** It took either a count or a rate and guessed which:
  `observed = successes / reps if successes > 1 else float(successes)`. For a
  nominal 95% study, one hit in four hundred replicates — a total failure of the
  estimator — is not greater than one, so it was read as a rate of 1.0, which
  sits inside the band, and the assertion passed.

  No heuristic can separate a count of 1 from a rate of 1.0, so counts and rates
  are now separate functions, and `assert_proportion` raises rather than guesses
  when handed a value outside `[0, 1]`.
