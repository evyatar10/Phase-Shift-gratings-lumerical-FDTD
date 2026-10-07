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
A read-only server inventory recovered the original S1 spatial profiles and
calibration logs. Their hashes were verified and printed calibration fits
independently reproduced. Recomputed spatial widths/sigma for all16 profiles and all three campaign
MACs match their logs exactly; final width19.38500618568187um. The current
raw campaign forward is incomplete and lacks its H5, so final transmission
remains a stored scalar pending a separately planned physical remeasurement.

An idle GPU alone is not a reason to dispatch. Parallelize only independent,
useful experiments with verified cross-cluster numerical parity, unique
outputs and a combined shared-seat budget. Prepare readers, provenance and
conditional inputs during the current jobs. Keep active runner/source hashes
unchanged, preserve all failed evidence, and never cancel or clean up to make
room without explicit authorization.


## Preparing a cheaper matched-field diagnostic

Source inspection supports one new nominal field forward plus one port2
adjoint as the minimum matched-duration pair; raw quadratures can then be
reconstructed from the same fields if the normalization/identity controls
pass. Separate Re and Im FDTD solves are not intrinsically necessary.
This is still a one-direction diagnostic, not global calibration.

The current100/150 ps runs deliberately have optimization_dft disabled.
They cannot supply the missing nominal E fields. A monitor-local subset of
frequencies may reduce DFT storage and extraction cost while retaining501
source/port samples. The three fixed bins alone do not cover the peak3
Jacobian; the previously selected union has five bins, which must be
recomputed from the actual new center or rejected if support changes.

A bounded hidden-CAD feasibility check is preparing this path during the
current solves. It will not solve FDTD or release another job. A custom
extractor would need exact recorded frequencies and supported Jacobian bins;
the vendor nearest-frequency lookup must not silently fill omitted bins.
Runtime savings cannot be predicted as501/5 because propagation remains.
See readiness_evidence_20261007/s2_matched_duration_fields_plan.md for the
source-level prerequisites, one-pair reuse limits and cost drivers.


Hidden-CAD preparation outcome: the exact custom five-frequency vector
persisted through save/reload while source/global/port501 settings stayed
unchanged. Four solve-call probes were blocked and four guard tests passed,
including parent rerun. Actual result count/grid remain untested. The first
attempt selected a pre-existing fake test scene and failed to load; the
preserved second attempt used the actual CAD scene and passed. All attempts
and source/input identities are in s2_selective_dft_cad_reviewed_1426.zip.
The inverse-design skill now records this qualified recorder guidance.

At14:18 IGUM101155 was still PENDING(Resources), and the shared solver pool
reported21/50 seats used by the three Athena jobs. IGUMsacct failed with a
connection-refused accounting-database error; queue visibility still works.
No dispatch or server reconfiguration followed that error.
