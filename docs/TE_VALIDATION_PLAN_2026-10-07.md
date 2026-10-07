# TE physical validation and runtime plan

Current16:33:35 FDTD complete; NEW fixed200ps Athena171390 tasks0/1/2
queued Priority after reviewed release16:26:12. Calibration remainsFAILED.
See TOP `TE_CONTINUATION_2026-10-07.md` for exact hashes/outputs and next steps.
N80 is first physical length candidate; N70 deferred; no length solve released.

## Update after completed 150 ps step-size comparison

**35 new FDTD solves are verified complete. Full optimization is not ready:**
global S2 gradient calibration remains failed. Athena171276 and CPU171300
are complete; the latter performed no FDTD solves. See the TOP of
[TE_CONTINUATION_2026-10-07.md](TE_CONTINUATION_2026-10-07.md) for exact jobs,
source/artifact hashes and the latest active preparation.

The h.0625/.125nm comparison still shows step-sensitive peak transmission
derivatives. The lead selected a new fixed200ps center/shift30+/-0.0625nm
triplet for preparation, to measure remaining temporal sensitivity; it has
not yet been released or submitted. No automatic further rung, adjoint,
mesh or N80 solve is approved by completion alone.

Existing100/150ps raw fields yielded spatial FWHM changes below.00085um;
this does not validate transmission gradients or mesh accuracy. The sameN98
field cropped toN80 changes its floor-relative FWHM by~.15um without clipping
halfheight crossings. PhysicalN80 remains unmeasured. A verified MAT product
and comparisonplot are under `results_from_athena/te_readiness_20261007/spatial_profiles_1558/`;
seven reader tests and parent exactJSON/MATpayload reproduction passed.
Currentplan keepsN98/free60/currentwidthbands, and uses bothclusters when
independent tasks justify it with current sharedseat/queue checks. Historical
Claude N80-too-short evidence belongs to a different TMcorr325family.

## IGUM completion verified 15:24 Israel

IGUM101155_0 finished15:20:01, COMPLETED exit0 confirmed by exact scontrol
array-task record, batch new-checkpoint/Finished/exit0 and successful solver
log. sacct still fails with database connection refused; that failure is
preserved, not interpreted as job state. **33 new FDTD solves verified complete.**
Athena171276 tasks0/1 remain running; latest15:20 engine42.08ps/150ps, about
36.5min engine remaining. No new checkpoint yet and no duplicate submission.

IGUM nominal checkpoint SHA fb42ab353a23af6991b9dcb261cb8d09c7c21267d06de63ecbebb546abd2449a;
ports SHA73e054409150ca72f6f19c6ab9ba004d1715d047fbe87f92541d04fd8cca3c06.
Original bundle SHA b675834acf8b7e303593ae5c62e727127c3cf70d05f26b0aae61733d828e0ba5.
Reader seven tests+parent and actual comparison passed. Preserved archive
`readiness_evidence_20261007/igum_qualification_verified_1524.zip` contains
original small artifacts, source/vendor hashes, full logs and comparison.

Against original-duration Athena center: exact501 grids/spec/296parameters/
frozen source hashes; compared installed API hashes equal. Both1010338steps,
67.39992ps auto-stop. IGUM-Athena peakTout=-1.2994393e-6, modal=+4.8933029e-8;
max absolute spectral difference1.3415369e-6 Tout/1.2235514e-6 modal. Same peak
bins/wavelengths. This supports nominal forward agreement across A10080/40GB,
not a repeatability bound or global gradient pass. Legacy CP lacks measured
saved CAD geometry, mesh coordinates and timing controls; these remain unknown.
Do not combine small FD legs across clusters as interchangeable measurements.
Global calibration/optimizer gate unchanged FAILED. Current100/150 profile
extraction and fixed-domain N80 preparation remain independent pending work.

## Mesh/length audit 15:20 Israel

See [TE_MESH_LENGTH_REVIEW_2026-10-07.md](TE_MESH_LENGTH_REVIEW_2026-10-07.md).
Parent reproduced all18 saved-envelope widths and the measured inventory
exactly; current S2 axes are49.106x50x49.3626nm. Historical mesh/box/length
evidence has useful but limited scope; neither50nm convergence nor a minimum
N is established for current S2. Preserve N98/free60/current bands. Existing
100/150 field_profile extraction is being prepared before new baseline solves.
N80 fixed-domain mapping preparation is separate and not released for solving.
The new convergence-and-length skill reference records these rules. Verified
text-only Claude Opus5.5 review was corrected against monitor-grid and objective
definitions; see reviewed archive1520. Focused commits66924a3 and1fbedb0 preserve
the running step pair and self-contained skills. Actual15:11 usage48%weekly
remaining; model-specific/Claude quotas unknown. No reset/purchase.

## Current update 15:08 Israel — step-size pair running; mesh/length audit added

Athena array **171276 tasks0/1** submitted **15:05:13 October7** and both
RUNNING on n307 at15:06, separate A100 PCI devices47:00/B7:00,8CPU/256G each,
2h limit, throttle2. Exactly shift30(index149)+/-0.125nm at150ps, reusing
verified150ps center task03. New runner `s2_temporal_step150.py` SHA
c41f522e8502598c4b8a3d4a4096baf094ede8a1c66763169992fbc82d696c8a.
Do not change this or its frozen dependencies while jobs/checkpoint readers
depend on them. No duplicate submission. Expected completion around16:00,
allowing50-65min per forward; actual150ps triplet elapsed53:10-53:36.

Release marker `/work/data/s2_step150_review_20261007_1507.json` SHA
8cc8e9da52262b19f4fd51d98c11de1a686c7e61b8c8bf2b364944922677dc0f.
Output `/work/results/diagnose_te_s2_step150/results/s2_shift30_step150_20261007`.
Seven offline groups, parent rerun, actual tasklist0/1, both hidden CAD geometry/
control/persistence checks and independent source review passed. Strict remote
parent checkpoint/FSP/H5/ports/log/source/vendor checks passed. Initial ad hoc
preparation omitted Linux config.LUMAPI_PATH and failed before any solve;
operator helper corrected to the existing wrapper config; runner unchanged.
Archive `s2_step150_prepared_verified_1506.zip` and dispatch1507 retain evidence.

Prepared strict completion fetch:
`C:/Users/evyat/AppData/Local/Temp/s2_step150_completion_fetch_f19beab1f86c/fetch_step150.py`,
SHA ebbe44a0aba524ed22a641b33287a31c5e88d2dc52bc43c7521a72ac0d6702c9.
Use actual job171276, exact marker/SHA above, existing
`readiness_evidence_20261007/s2_step150_fetch_env_1509.json`, NEW absolute output.
Both tasks must complete exit0 before fetch; read-only cold validation blocks
main/CAD/solve. Strict local same150ps h=.0625/.125 analyzer is being prepared.

IGUM **101155_0 now RUNNING**, started14:49:19, A100-SXM4-80GB ece-efrats3,
8threads verified. It is the original-duration center qualification, not the
150ps family. Do not pool its derivatives before parity review. Earlier IGUM
accounting DB unavailable; latestqueue/log/scontrol valid. At15:03 shared seats
7/50; new Athena adds14 for21 total, quota228G/300G. These are dated snapshots.

32 new FDTD solves verified completed; current jobs are NOT counted completed.
All global calibration gates remain FAILED. No coefficient/optimizer/200ps/
new adjoint release. Offline alternative spline extraction shifts100/150
peak derivatives4-10%, insufficient to explain own-J discrepancy; synthetic
translation only illustrates estimator sensitivity, not actual error/noise.
Parent reproduced exact report and3tests passed; archive
`s2_peak_extraction_verified_1506.zip`. Objective unchanged.

User additionally requests mesh(~50nm) and N~100 spatial-width convergence and
shorter-length feasibility. Focused audit recovered raw historical MAT/workbook
evidence; report pending. Keep currentN98/free60/mesh while checking. Separate
temporal, mesh, longitudinal length and transverse box tests; legacy uniform
evidence cannot by itself qualify current nonuniform seed or physical gradients.

All13 project Codex skills now bundle byte-preserved detailed runbooks locally;
18personal packages/306files verified. Claude originals unchanged; current
adaptations/rules take precedence. Parent all13 validator/hash checks passed.
See CLAUDE_SKILL_TRANSFER_2026-10-07.md and selfcontained inventory/validation.
Latest actual usage14:56 weekly51% remaining; other/model-specific pools unknown.
Latest committed milestone591db93; current diagnostic/skills checkpoint pending.

## Decision after verified150 ps,14:46 Israel

All150ps tasks completed and cold verification/strict corrected reader passed.
Count32. The detailed comparison is s2_temporal150_review_1442.json. Peak
transmission changed modestly but derivative changes100→150 remain material;
no convergence/global calibration pass. Next prepare and qualify exactly two
150ps forwards at shift30±0.125nm, reusing the existing150ps center. Compare
both step sizes with movingpeak and own-center-Jacobian/fixed-bin analyses.
This tests finite-step behavior at fixed duration; it does not automatically
select a trustworthy derivative or make150ps adequate. Runtime budget50-65min
per job,2A100/14shared seats after fresh preflight. No dispatch yet.

In parallel, the five-frequency CAD preparation is complete, while a bounded
offline peak-estimator sensitivity analysis examines the existing spectra.
A200ps family and field-enabled control remain conditional future choices.
Original temporal sources remain frozen, even though jobs have completed.
The following sections preserve the advance plan and earlier estimates.

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
