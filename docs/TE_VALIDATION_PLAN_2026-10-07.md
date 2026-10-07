# TE physical validation and runtime plan

Updated October 7, 14:11 Israel. Full inverse-design readiness remains **NO**.
See [the evidence map](TE_VALIDATION_EVIDENCE_MAP_2026-10-07.md) for what the
original Claude checks establish, what was independently reproduced, and
which raw artifacts or physical checks remain missing. A fit or an offline
pass is not an independent physical-gradient qualification.

## Current decision and critical path

The immediate question is whether the S2 shift30 finite differences are
stable against solver duration at the same geometry, perturbation and grid.
The approximately68 ps to100 ps comparison changed both transmission spectra
and peak gradients materially. That is measured temporal sensitivity; it
has not established the cause of the failed global adjoint fit.

The critical path is:

150 ps triplet -> strict artifact/control checks -> full temporal comparison
-> review the next forward-consistency experiment -> matched forward/adjoint
and held-out parameter-class validation -> moving-resonance spatial-width
qualification -> bounded current-driver N98 optimizer smoke.

A full optimization campaign still requires separate authorization. None of
these arrows releases a dependent simulation automatically on exit0.

## Running work and expected cost

Athena171236 tasks3/4/5 are the nominal center and shift30 +/-0.0625 nm at
150 ps. They started13:40:15 on threeA100 GPUs,8CPU/256G each,2h limit.
At14:10 all three were RUNNING, with81.1-82.6 ps simulated and about23 minutes
of engine time remaining. No completed checkpoints existed. Estimated output
availability is14:33-14:40, allowing extraction/verification; this is an
estimate, not a reservation or guaranteed completion time.

The comparable100 ps jobs took35:45-35:51 wall time each. An advance150 ps
budget of50-65 minutes per job is reasonable for this measured hardware and
family: roughly2.5-3.25 GPU-hours total,50-65 minutes elapsed when all three
start together, plus queue and subsequent analysis. Actual logs refine this
estimate. Auto-stop jobs, adjoints with large field recording, and different
GPUs cannot inherit this estimate without accounting for their overhead.

Last dispatch preflight: Athena226G/300G soft quota; shared license pool0/50
used before launch, new Athena demand21 seats plus queued IGUM commitment7.
That snapshot is historical, not current spare capacity. Recheck both queues,
quota, seats and limits before any new solve. IGUM101155 qualification remains
pending in the latest inspected snapshot; do not duplicate it.

Saved-field CPU Re/Im reconstructions took8:21/8:24 and no new FDTD solves.
Their128G allocations were serialized within240G QOS memory. Reuse those
completed controls instead of rerunning them. Current completed FDTD count29;
three running jobs are excluded until completion and result verification.

## Prepared result handling

The strict150 ps fetch helper is already tested (six groups), and the strict
100+150 ps analyzer is already tested (seven groups plus actual100 ps inputs).
Exact launch, marker, source, helper and checkpoint identities are retained in
TE_CONTINUATION_2026-10-07.md and the dated evidence records. Fetch original
bytes to a new directory; independently record the new pin-map hash. Require
all tasks COMPLETED/exit0, actual150 ps controls/status, original grids and
frozen source identity. Do not call a solve entry point during extraction.

Compare full501-point spectra, both Tout and modal S21 power, center values,
central odd derivatives, even terms using each duration's own center, moving
peak3 objectives and fixed nominal-Jacobian projections. Keep same-duration
and historical Jacobian projections separate. Do not mistake spectral FWHM
for the spatial design constraint. Old adjoint fields use old stopping
conditions; fitting them against150 ps FD is not matched calibration.

## Conditional next experiments

| Evidence after150 ps | Next useful action | Cost / release boundary |
|---|---|---|
| Failed artifacts, wrong duration or changed identity | Diagnose logs and preserved artifacts; no coefficient fit | Read-only / CPU first; do not repeat a completed solve blindly |
| Duration changes remain material for the derivative decision | Select a justified longer-duration family or targeted temporal diagnosis | Reuse prior evidence; review a minimal center/plus/minus group and its runtime before dispatch |
| Duration comparison is sufficiently stable for a bounded decision | Check finite-step consistency at the chosen duration, reusing its center | Minimal symmetric pairs, not a full new calibration matrix; inspect peak branch movement |
| Reliable directional FD but mismatch to reconstructed operator persists | Match source/objective/normalization and actual simulation family; obtain needed new forward/adjoint fields | Include field-storage, adjoint and CPU contraction cost; old fields cannot certify changed termination |
| Representative classes and independent held-out directions pass | Qualify width gradient and total moving-resonance width prediction at the actual starting point | Separate width source/twin as required; do not transfer another seed or gate point |
| Scientific gates pass | Run the authorized bounded N98 hardware smoke on the integrated code | Actual accepted/rejected rows, transmission, spatial width, restart and budget behavior |

Preserve the existing10% per-class/held-out calibration criteria for resolved
sensitivities. For cancellation-dominated or nearly zero derivatives report
absolute as well as relative discrepancies and their conditioning; do not
invent a noise floor or silently relax a gate. A pair of durations agreeing
is limited evidence, not a general convergence certificate.

## Useful parallel work

The historical validation evidence map is now prepared. Independent100 ps
spectral-width and normalization arithmetic has passed its own tests and a
parent reproduction. Rules/skills now require this runtime-aware planning.
A focused S1 replay assessment is checking whether existing raw files can
answer the original-results question before requesting another solve.

An idle GPU alone is not a reason to dispatch. Parallelize only independent,
useful experiments with verified cross-cluster numerical parity, unique
outputs and a combined shared-seat budget. Prepare readers, provenance and
conditional inputs during the current jobs. Keep active runner/source hashes
unchanged, preserve all failed evidence, and never cancel or clean up to make
room without explicit authorization.
