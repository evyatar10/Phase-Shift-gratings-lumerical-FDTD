# Full-optimization readiness criteria

This is the completion contract for the authorized overnight preparation.
It is not permission to launch a full optimization campaign. A passing offline
suite or N70 pipeline smoke alone cannot establish N98 physical readiness.

## Required before a full campaign

1. **Current-objective gradient evidence.** Complete S2's twelve FD legs and
   peak3 quadrature reconstructions, inspect full-precision class/sign/held-out
   errors against the existing 10% criterion, and pin the measured calibration
   with its provenance. Do not borrow S1 coefficients. S1 needs its own
   current-objective evidence before restarting that lane; historical softmax
   calibration is not interchangeable.
2. **Reliable optimizer and restart.** Fix confirmed acceptance-reference,
   stale-measurement, missing-width repetition, missing-wavelength-gradient,
   profile-identity, journal/state recovery and final-result pairing bugs.
   Test real driver paths with negative controls; preserve production sources
   while the recovery chain fingerprints them. Enforce the global forward
   budget across recenter retries and cold resumes.
3. **Hardware execution.** After calibration passes, integrate the isolated
   candidate and run the bounded N70 smoke. Inspect measured metrics, profiles,
   acceptance decisions and cold-resume behavior. No missing required metric
   may silently become a successful optimization result.
4. **N98 local physical validation.** Exercise the retained N98/free60/bare
   device before claiming full readiness: compare actual trial width and
   wavelength changes with their local predictions, and inspect both modal
   transmission and T_out objective. The S2 field near-zero component's 23.8%
   relative error and S1 historical held-out exception remain qualified
   evidence, not unconditional passes. Use bounded meaningful directional
   checks rather than launching a long campaign to discover these failures.
5. **Explicit width and transmission contract.** The latest clarification is
   nearly constant spatial FWHM; retain the seed-specific ±2% bands. An
   optional seed-width upper bound is available if literal no growth is later
   chosen, but do not silently tighten existing bands. Enforce the same hard
   bounds in measured acceptance and restart/final selection as in proposals.
   Do not call historical S1's +1.38% width growth exactly unchanged. Keep
   abs(T_out) peak3 and modal abs(S21)^2 distinct. Validate the opt-in measured
   modal guard and modal final selection in a bounded pilot; its thirteen
   offline suites have passed, hardware is pending. The calibrated T_out
   derivative remains a proxy, not an exact modal gradient or a universal
   rescaling. See `TE_OBJECTIVE_DECISION_2026-10-07.md` for the local wavelength
   safeguard, conditional reuse and separately gated combined-source AL.
6. **Runnable, bounded launch packet.** Verified seed/vector/bounds, numerical
   family, accepted-state provenance, calibration hashes, resource and license
   checks, explicit global evaluation limit and fresh output identity. Include
   exact launch and resumption commands and no overlapping writers. The user's
   full-campaign authorization is still a separate step.

## Additional diagnostics that must remain visible

Record mode identity/MAC, peak location within the scan, spectral sampling and
linewidth/Q, reflection and residual 1−T−R, profile extraction validity,
moving-width residual, wavelength-proxy residual, twin wavelength lag,
trust-region predictions/actual gains and termination reason. These quantities
help identify false progress; they are not silently added optimization targets.
Domain/PML/mesh convergence and model-derived intrinsic Q remain distinct
from an optimizer regression test or a fixed-grid improvement.

## Current limits

06:43 update: center repeat170901_0 COMPLETED and explicitly reviewed.
Peak3(Tout) repeats within4.751e-7, identical spectral grids; one repeat does
not establish a noise distribution or gradient accuracy. corr30+/-0.5nm
diagnostic array170906 is active, serially, after a SHA-pinned center review.
All calibration and optimizer hardware gates below remain FAILED/BLOCKED.
Actual account query06:40 now reports100% remaining in its listed weekly
codex bucket; change cause unknown, and no reset redeemed by this workflow.

At 06:02 Israel time, all twelve S2 FD legs and both peak quadratures are
complete. Both legacy and current peak3 calibration FAIL the 10% gate;
neither can be installed. Independent minimax fitting still leaves58.1%
worst relative error. Candidate integration and all13 actual offline gates
are complete; optimizer hardware smoke and N98 validation remain blocked.
Forward-only nominal-repeat diagnostic170901_0 is running; it is not an
optimizer or calibration pass. Read `TE_S2_CALIBRATION_FAILURE_2026-10-07.md`.
Historical S1 final spatial profiles and raw spectra
are not available in the local result bundle. See the saved-results audit
for exactly which measurements were reconstructed versus read from logs.

The native thread heartbeat is active; the old recovery watcher exited on
the failed fit as designed. The heartbeat now monitors the diagnostic. Its
existence does not guarantee model availability after usage exhaustion. The
account reported6% weekly Codex allowance remaining at 05:49 Israel time,
reset October12 at10:44 Israel time. There is no defensible conversion of this
percentage into guaranteed research hours or a number of tomorrow's runs.

Primary evidence: `TE_CONTINUATION_2026-10-07.md`,
`TE_RECOVERY_2026-10-06.md`, `TE_ALGORITHM_AUDIT_2026-10-07.md`, and
`TE_SAVED_RESULTS_REVIEW_2026-10-07.md`.
