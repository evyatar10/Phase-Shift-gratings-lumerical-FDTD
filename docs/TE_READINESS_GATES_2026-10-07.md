# Full-optimization readiness criteria

**09:03 access blocker:** both clusters timed out; local VPN adapter is disconnected. Last verified170962 running08:57, IGUM101155pending08:54. New five-class no-solve helper is locally tested but NOT uploaded/dispatched. See current continuation for exact resume steps. Failed scientific gates unchanged.

This is the completion contract for the authorized overnight preparation.
It is not permission to launch a full optimization campaign. A passing offline
suite or N70 pipeline smoke alone cannot establish N98 physical readiness.

## Latest diagnostic status — calibration still blocking

Athena170926_0–3 completed; all18 NEW overnight FDTD solves are complete.
The role-aware corr30 four-step analysis passed six tests and actual evidence
checks, with h2,1,.5,.25nm derivatives1.8748649e-5,3.6950366e-5,3.6672609e-5,
3.4911852e-5 pernm. This is step-dependent evidence, not an established error
order or noise bound.

Stored-field Re170950_0 and Im170956_0 completed without FDTD solves. Their
h1 contractions match original reconstructions near roundoff; smaller dEps
steps change raw quadratures by less than0.1%. The old failed-fit combination
is stable but unvalidated/uninstalled. Narrow provenance/extraction/mesher
checks do not satisfy the unchanged10% per-class and held-out gradient gate.
See `readiness_evidence_20261007/s2_corr30_four_steps_summary.json` and
`s2_deps_both_review.json`.
The server probes measured PVA `meshing refinement`5 in all inspected contexts.
No mesh-refinement setting was changed to obtain these results.

A staged h0.5nm class runner is prepared and passed six offline groups plus
hidden CAD task2 with zero local FDTD solves. First-wave0–3 repeats corr30 and
measures shift30. Athena170962 array0–3%4 was submitted08:55:24 with
A100/8CPUs/256G/2h/12h_4g. At08:57 all four tasks were RUNNING with
8thread/GPU startup readback; no results are claimed yet. Remote no-CAD
identity validation passed. Eighteen completed solves remain unchanged;
these four new forwards and queued IGUM are outstanding, not completed.
Remaining4–11 require a separate lead-reviewed marker with authentic first-
wave checkpoint SHAs and verdict. Preparation is not approval for later legs
or for optimization. IGUM101155_0 remained queued at08:54; refresh resource
and shared-seat checks before later release. At08:54 shared occupancy was0/50,
with35/50 seats committed including IGUM and the new array; quota230G/300G
soft/330G hard. The hardware optimizer smoke and N98
physical-readiness criteria below remain blocked by failed calibration.

## Historical diagnostic status, 08:03 Israel

Failed S2 calibration remains blocking. Four corr30 forward-only convergence
checks170926_0–3 are RUNNING; IGUM baseline101155_0 is queued. No-solve
geometry/grid provenance170921 passed its narrow consistency checks, not the
gradient gate. Do not repeat it or promote one corr30 result to global readiness.

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

07:54 update: corr30h0.5nm pair completed; its measured derivative is95.6%
larger than the h2nm derivative, so finite-step dependence is material.
No-solve Athena170921 completed exit0: all723 mapped geometry properties
agree with te_point to1.7e-21m, saved forward/adjoint field coordinate arrays
match exactly, generated/index axes differ only at floating-point scale.
The adjoint port's49.106nm offset is intentional in inspected vendor code.
This removes support for gross geometry/grid mismatch in the checked data;
it does not pass the full gradient gate. IGUM101155 baseline qualification
is queued for resources. Full optimization remains blocked.

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
