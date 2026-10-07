# S2 calibration failure and bounded diagnosis

## Update14:46 Israel -150ps completed; next150ps step pair being prepared

Athena171236 tasks3/4/5 COMPLETED exit0. Cold artifact checks passed for exact
150ps/status1/2248530iterations, frozen runner555f and all FSP/H5/NPZ/log SHA.
Completed new FDTD count32. Original files are in
readiness_evidence_20261007/s2_temporal150_fetch_1434; pinmap SHA
4fbe923b0fb770b19670946773edc5f892b0efedd07d821a3fee2a750793670b.
Reviewed archive s2_temporal150_verified_1445.zip preserves originals and reader.

The initial strict local reader rejected a2ULP Port_1 phase difference between
100/150 before-control snapshots for center/minus (plus exact); no other
nongeometry control differed. An isolated reader accepts ONLY that measured
<=2ULP phase difference, retains raw values, keeps other checks unchanged and
does not localize its cause. Seven new plus seven inherited test groups and
parent actual rerun passed. New reader TEMP s2_temporal_phase_fix_6cf50c5e45e2,
SHA bbe3a22f90a418c8cadc3fc0c7b95f8843b4d83127d611bb0f4481bb7fdee3dd.
Frozen runner/original reader unchanged. Full report s2_temporal150_review_1442.json,
full spectra s2_temporal150_spectra_1442.json, compact summary1445.

Tout peak3 center100→150 .933286528→.933570417; modal .979764300→.979971606.
At h=.0625nm Tout movingpeak FD -.000208331→-.000153815 /nm; modal
-.000316571→-.000163699 /nm. Same-duration fixed-J projections at150 are
-.000448383 Tout and+.000333596 modal; plus/minus peaks move across bins.
Temporal/finite-step/selector sensitivity remains, not convergence or an
adjoint calibration pass. All global failed gates remain; no coefficient or
optimizer release. Old fields still belong to old termination settings.

Lead chooses next bounded150ps shift30±.125nm pair, reusing exact150 center,
to separate step sensitivity at fixed duration. NEW isolated runner/tests are
being prepared by objective_code_audit in TEMP; no upload/dispatch yet.
It must bind center CP SHA a0ff55eacaa52e00a2aab17dcc43d0e5fff9a72566e15da371ff25279c172663,
all150 checkpoints, frozen dependencies and substantive review marker, followed
by hidden CAD, independent review and fresh seats/queue/quota. Expected50-65min
per forward once started,2GPUs. No200ps or new adjoint run released.

Verified text-only Claude Opus5.5 reviewed next-stage planning; its assertion
that the existing150runner could change h without newcode was corrected.
Large even terms or two disagreeing h values do not establish discreteness,
a noise floor or automatic preference for larger h. Separate offline peak
extraction sensitivity analysis is underway; do not silently change the FOM.
Current conditional plan TE_VALIDATION_PLAN_2026-10-07.md. Lastusage14:33 reports
55%weekly remaining; other/model-specific pools unknown. Latestcommit ba9db76
preceded this completion. User requested sustained physical validation and
runtime-aware planning; keep source/results safe and continue routine steps.

## Current state at 13:47 Israel — temporal validation continues

The global S2 calibration remains **failed**. Neither coefficients nor an
optimizer campaign have been released. The current continuation is recorded
at the top of `TE_CONTINUATION_2026-10-07.md`; older sections below are history.

All three 100 ps tasks in Athena array171216 completed and passed artifact,
control, exact-grid and full-duration checks. The completed FDTD count is29.
The change from the earlier approximately68 ps family materially changed both
modal and outgoing transmission and their finite-difference peak gradients.
See `readiness_evidence_20261007/s2_temporal100_review_1339.json` and summary1341.
This establishes temporal sensitivity, not infinite-time convergence or a
repair of the original calibration. Original adjoint fields remain tied to
their original stopping conditions.

After explicit review, the same center/shift30±0.0625 nm triplet at150 ps was
released as **171236 tasks3–5** at13:40. It is running, not complete. Frozen
runner555f and separate150 ps marker6cc6d9fd...527f51 remain unchanged.
Both old-field CPU quadrature reconstructions171220/171225 completed; their
successful arithmetic controls do not pass the physical calibration gate.
Latest evidence commit is `0ded829`. No full optimization is authorized.

## Update13:05 Israel - staged temporal forwards and corrected CPU reconstruction active

Athena **171216 tasks0,1,2** submitted13:01:23 for exactcenter/shift30+/-0.0625nm
at common100ps. Each requestsA100,8CPU256G,2h,12h_4g;maximum3concurrent.
Allthree RUNNING13:02 on n307/n309/n312; GPU and8-thread setup readbacks passed.
Runner SHA555f3d13b42c494171e7658cc4263c588ce418e5d937da1497d690824b41d44a;
marker /work/data/s2_temporal100_review_20261007_1257.json SHA
951c2025bf3255830a3b13e03a333b18bfb0ba59c86ee4e1c2b150ac078e2206.
Output /work/results/diagnose_te_s2_temporal/results/s2_shift30_temporal_20261007.
Twelveoffline groups passed including actual main(task_idx=...) dispatch; earlier
keyword-interface mismatch was fixed before submission, remote6e7 source preserved.
Finalcontrols unchanged from tested hiddenCADsave/reload; actual100ps termination,
status1, exactgrids and retainedartifacts still need hardwareverification.
150ps tasks3..5 remain blocked by separate reviewed100ps-result marker.
No new simulation is counted completed until its full evidence is checked.

CPU-only corrected **171220_0 Re** submitted13:03:47,8CPU128G45min2h_2g,
output /work/results/s2_fixed_bin_v3_re_20261007T1305/fixed_bin_shift30.json.
Probe0c9a078618862ff032fc41158575e3e66fc1d3c65bd5f14c7044ccd3bfcf7b34,
aux d73ad934bb2e58c0ff4fea27d1904d9b357c1fd974fe09db5798d480f4f1ed45.
Tenfocused tests andparentrerun passed. Full501 metadata snapshots retained;
exactsource/frequency/power,32eps-relativecomplexamplitude arithmeticreuseonly.
Originalpeak3control unchanged. Inspect newRe result andpreservation beforeIm.
PreparedIm wrapper paths in s2_fixed_bin_v3_wrapper_preparation.json; notuploaded.
Allv1failedfiles and metadata-only171210 evidence remain preserved.

Fresh12:57 shared0/50used,IGUM101155 pendingResources; newAthena21+queuedIGUM7
bounded28/50.13:03Athenaquota224G/300Gsoft. CPUadds0FDTDseats. Sharedseat
snapshot is not a reservation; recheckbeforefuturedispatch. Lastusage12:39weekly
70percentremaining, separatepoolsunknown. LatestGit5e86684; thisnewstagependingcommit.
Sol provenanceworker prepares strict temporalresultpostprocessing during the wait.
Globalcalibration remains FAILED; nooptimizer/coefficient/productionobjectivechange.

## Update12:22 Israel - smaller-step ladder complete; peak derivative still unstable

**26 new FDTD solves verified completed.** All171162 tasks COMPLETED exit0,
remote cold-resume validated4 retained checkpoints/FSP/H5/solver logs/NPZs;
original bytes fetched and independently pinned. Actual strict analyzer passed
exit0 with exact grids/source identities/scores. Artifact-pin SHA
1d9b1dbf10ffa14dc11d758af4fe88c5ae2d0e579145b027711f3bb9ee124692.
Reports: readiness_evidence_20261007/s2_shift30_measured_1219.json,
s2_shift30_spectra_1219.json and compact s2_shift30_summary_1222.json.

Tout peak3 central derivative at h2,.5,.125,.0625nm is respectively
-1.57680e-4,-2.70328e-4,+6.68587e-4,-1.76330e-3 pernm. Modal at the new
steps is+5.55795e-4 and-2.03270e-3. Tout even terms remain2.45462e-4 and
5.01823e-4 at the two new steps. This is NOT a reliable peak-gradient plateau;
smaller steps alone did not resolve calibration. Frozen-center-J projections
at the new steps are+8.10773e-4 and-1.57975e-3(Tout), also unstable.
Fixed flank-bin244/260 Tout derivatives change~1.1-1.4% between newsteps,
while the small central/peak difference changes much more. This is local
observed behavior, not a global convergence/noise bound or isolated cause.
Remaining8otherclasslegs unreleased; no tighter/new FDTD array now.

**CPU-only Re fixed-frequency diagnostic171195_0 submitted12:20:50**:
8CPU128G45min2h_2g/l40s-public, zeroGPU/newFDTD, output
/work/results/s2_fixed_bin_shift30_re_20261007T1216/fixed_bin_shift30.json.
Helper SHA b5f1aeef94d9c1d08ea34e0c265e1e0e1b8ec6ec7f3adf0c4ff63bc03fab23c4;
frozen mechanics0be8299d...d526d and guard40dd4878...99d unchanged.
Five tests and parent rerun passed; direction-aware source metadata corrected
and tested. It first reproduces original shift30 peak3 h1, then computes raw
one-hot244/252/260 using five cached wavelengths and unchanged vendor scaling.
Inspect actual metadata/control/preservation before serial Im. No Imdispatch yet.
Source package: s2_fixed_bin_prepared_1220.zip; receipt s2_fixed_bin_re_dispatch_1220.json.
Preflight12:14 shared7/50 used plusqueuedIGUM7; CPUaddsnoFDTD seats. Athena
queueempty12:20,quota224G. IGUM101155 stillpendingResources12:14.

Sol s2_gradient_provenance_audit is inspecting nine actual termination logs
(s2_diagnostic_termination_logs_1222.json) for temporal-convergence hypotheses.
No claimed causality: frozen-grid discretization and finite-timeDFT remain
possibilities. Source/gradient calibration, optimizer smoke and N98 width gates
remain FAILED/BLOCKED. No coefficient or production objective change.

## Decision sequence after the smaller shift30 ladder

1. Verify completed remote artifacts before interpreting any new scalar. Compare
   fixed-frequency FD arrays across h, moving-peak FD and both nominal-J projections;
   retain both-center evens. Stable-looking values are local evidence, not a global
   convergence/noise guarantee. Peak-bin changes alone do not prove a bad adjoint.
2. Use the separately prepared saved-field one-hot reconstruction, if its exact
   source/normalization metadata, finite slices and old peak3 control pass. Compare
   raw quadratures to fixed-frequency Tout FD, with modal FD shown separately.
   Do not silently identify the implemented source-power operator with modal S21.
3. If fixed-frequency forward derivatives stabilize but the corresponding operator
   cannot predict them, investigate source/phase/normalization and mesher derivative
   accuracy. This does not uniquely isolate one cause. If fixed-frequency FD itself
   remains step-sensitive, objective smoothing alone cannot establish calibration.
4. Consider replacing the peak estimator only after this operator distinction is
   understood. The quartic model improvement is conditional and has bias/mixed
   modal behavior. No failed coefficients, automatic remaining-class release or
   optimizer are authorized by these diagnostic comparisons.

Source/field feasibility review and corrected independent consultation are saved
in readiness_evidence_20261007/s2_stored_field_reweight_reviewed_1158.zip and
claude_next_decision_review_1148.json. Production numerical family remains frozen.

## Update11:42 Israel - four smaller shift30 solves submitted

Athena array **171162 tasks0-3** submitted11:41:31 for shift30(index149)
+/-0.125nm and +/-0.0625nm, one forward per task, original recovered center.
These are diagnostic step choices, not certified converged gradients.
Runner SHA7f0500a8a037961a3d72e3aa5d58ae2cd52ebbab2a8505f5395097adfac5c857;
new immutable lead marker SHA50ef62531e338f57b5cbc78d05905a2cd5d805297407a4ba0913cedf3a5404df.
Five focused offline tests and hidden CAD +0.0625nm passed; all723 geometry
properties match, locked grids unchanged. Remote template/marker validation
reverified completed first-wave checkpoints, spectra and preserved artifacts.
Original dependencies/markers remain frozen; no release of eight other class legs.

Each task requests A100,8CPUs,256G,2h,12h_4g; maximum4 concurrent.
Preflight11:39: shared solve seats0/50, queued IGUM101155 commits7, new
Athena commits28(total35/50); quota227G/300Gsoft/330Ghard; Athenaqueueempty.
IGUM baseline remains PENDING Resources. Submission is not a completed solve:
**22 completed new FDTD solves**, plus4 newly submitted and1 IGUMpending.
Dry-run estimated12:21 start, not guaranteed. Inspect actual queue/startup next.
Receipt/environment/original marker: readiness_evidence_20261007/s2_shift30_*.
Expected root /work/results/diagnose_te_s2_shift30_convergence/results/s2_shift30_convergence_20261007.

Isolated estimator comparison: quadratic7 not promising; quartic11 reduces
cubic-model Tout local derivative ripple~29x but has~9.4e-6 height bias,
mixed modal behavior and strong interpolant dependence. Neither removes
actual shift30 even term~2.8-3.0e-4. Five tests and pinned actual-data analysis
pass. Production objective unchanged; saved scalar Re/Im cannot directly
supply7/11-sample Jacobians. Stored-field/source compatibility is under bounded
review before any such reconstruction. Models are not physical calibration.
Evidence: s2_peak_estimators_reviewed_1142.zip and s2_shift30_prepared_tests_1142.zip.
Global calibration remains FAILED; no coefficient installed and no optimizer.

## First-wave measured evidence and next check, 11:18 Israel

**Global calibration remains FAILED; no coefficients are installed.** Access
returned at11:12. All four170962 tasks completed around09:27, exit0, and
their downloaded original checkpoints/NPZs passed full identity, grid and
scalar-reproduction checks. Remote cold resume reused the same four results
and checked retained artifacts. The completed NEW FDTD count is22. These
checks establish evidence fidelity, not scientific calibration accuracy.

| h0.5nm measurement | Previous derivative pernm | New derivative pernm | Observed change |
|---|---:|---:|---:|
| corr30 peak3(Tout) repeat | +3.667260923146e-5 | +3.694982916536e-5 | +0.75593% |
| corr30 modal peak3 repeat | +4.408675869771e-5 | +4.177987100784e-5 | -5.23261% |
| shift30 peak3(Tout), old h2 versus new h0.5 | -1.576796279270e-4 | -2.703278141175e-4 | Step sizes differ |

shift30 Tout endpoint indices are241/263 around center252:11 bins each way.
Its h0.5nm exact fixed-center spectral-Jacobian FD projection is
+3.858571312957e-4/nm, opposite to the moving peak's finite-step derivative.
This is not proof the infinitesimal observables differ or that the adjoint is
wrong. Two corr30 pairs do not estimate a statistical noise distribution;
two shift30 sizes do not establish an asymptotic error order. Preserve all
endpoint spectra and both old/new centers. Evidence:
`readiness_evidence_20261007/s2_firstwave_measured_1113.json` and
`s2_firstwave_spectra_1113.json`.

The lead explicitly holds the remaining eight h0.5nm class legs. Do not
extrapolate the corr30 result blindly or publish a later-stage marker.
Instead CPU-only Re171145_0 was submitted11:17:22 to reproduce the five other
classes' original h1 scalar contractions: indices0,60,120,149,295. Resources
128G,8CPUs,45min,2h_2g/l40s-public,zeroGPU/noFDTD. Source SHA
`0be8299d6607388f373876bf16b5f15e3d894220efcd770b9cccedd6f46d526d`;
output `/work/results/s2_other_classes_h1_re_20261007T0901/other_classes_h1.json`.
Inspect Re fidelity and preserved artifacts before serial Im. Receipt:
`readiness_evidence_20261007/s2_other_classes_re_dispatch_1118.json`.

Claude5.5's advisory response is saved in `claude_firstwave_review_1118.json`.
The lead rejects its unsupported O(delta^2) peak-ripple cause claim pending
the bounded TEMP-only current-objective code/model test. Available recovered
adjoint outputs are scalar objective contractions, not a full exported
wavelength-resolved adjoint Jacobian. No claim of objective failure, repaired
gradient or passed calibration is adopted before those gates.

The existing heartbeat resumed ACTIVE11:18. IGUM101155_0 is live-queue pending
Resources11:15; its11:12 accounting DB error is not lost SSH access. Shared
preflight pool0/50,queued IGUM commitment7; the CPU check adds no solve seats.
Original source/result history and the unchanged10% calibration gate remain.

## Latest four-step and both-quadrature evidence

**Calibration remains FAILED; no coefficient is installed.** Athena170926
tasks0–3 completed the h1 andh0.25nm corr30 forward pairs. The role-aware
analyzer passed six tests and actual evidence validation, preserving old/new
centers, spectral grids, peak scores and distinct historical source roles.
The h2 center hash identifies a read-only export, not original FSP creation;
its endpoints have separately pinned recovery-generated identities.

| corr30 FD half-step nm | peak3(Tout) central derivative per nm | Even term, repeated center |
|---:|---:|---:|
| 2 | 1.874864880122e-5 | +1.801618297168e-4 |
| 1 | 3.695036628504e-5 | +7.365857889552e-5 |
| 0.5 | 3.667260923146e-5 | -1.204750627926e-6 |
| 0.25 | 3.491185204285e-5 | -1.212745799783e-6 |

All Tout endpoint peak indices are252. Full spectra, exact center-Jacobian
projections and both-center even terms are retained in
`readiness_evidence_20261007/s2_corr30_four_steps_summary.json` and the full
measured report. The published verdict is diagnostic only. Halving differences
do not establish convergence order, solver-noise bounds or a diagnosed mesh
defect. h0.5 is the lead's next bounded comparison step, not a proven exact
infinitesimal derivative or a rescue of the global fit.

No-solve stored-field jobs170950_0 (Re) and170956_0 (Im) completed in roughly
five minutes each. Their h1 raw corr30 contractions match the original peak
reconstructions to4.108e-14 and5.248e-16 relative, respectively. Raw Re/Im
vary by less than0.1% over dEps steps1,.5,.25nm. The old FAILED-fit weighted
combination is4.9142174e-5,4.9143013e-5,4.9141353e-5 pernm: within0.00171%
of h1 and still different from the measured FD derivatives. These coefficients
are used ONLY to expose cancellation sensitivity; they remain uninstalled.
No coefficient was fitted by either probe. Both originals are preserved.
Evidence: `readiness_evidence_20261007/s2_deps_both_review.json`.
Raw Re/Im reports both read PVA `meshing refinement`5 in generated, original
saved forward/adjoint and nominal-index contexts. It was measured, not inferred
from the documented default, and no refinement setting changed.

The no-solve Jacobian is evaluated at each ORIGINAL saved forward; FD even
terms use the later repeated nominal center. This distinction is intentional.
The probes validate extraction/subset fidelity and narrow dEps-step sensitivity,
not every class, held-out prediction, spatial-width accuracy or optimizer
readiness. Original FSP hashes and HDF5 stats were preserved; HDF5 content was
not rehashed. Eighteen NEW overnight FDTD solves are complete; these two CPU
probes add zero solves. IGUM101155_0 remained queued in the08:45 snapshot.

Next staged forward-only check is prepared under
`lumopt2_te_s2_fdhalf_classcheck_20261007`: first corr30 repeat and shift30
pairs at+/-0.5nm, then only after explicit first-wave review the corr1,avg1,
shift1 and cavity pairs. Separate immutable stage markers preserve first-wave
checkpoint SHAs and a lead verdict. Live cavity index is295, not180. Six
offline groups and an actual hidden CAD shift30 build passed; no local FDTD
solve ran. Athena170962 array0–3%4 was submitted08:55:24 for only the first
wave (A100,8CPUs,256G,2h,12h_4g). At08:57 all four tasks were RUNNING,
with8thread/GPU startup readback; no result is claimed. This adds four
outstanding forwards, not completed simulations. Remote no-CAD identity checks
passed, and existing sources remained untouched. Receipt:
`readiness_evidence_20261007/s2_half_classes_first_dispatch.json`.
Startup: `readiness_evidence_20261007/s2_half_classes_startup_0857.json`.
The original failed10%
class/held-out gate, optimizer smoke and full-campaign restrictions are unchanged.

## Measured smaller-step and provenance results, 07:54 Israel

Athena170906_1/2 completed exit0. Exact source/center identities and wavelength
grids matched; exported spectra reproduced saved peak scores. corr30 h0.5nm
peak3(Tout) derivative=3.667260923146e-5/nm versus h2nm1.874864880122e-5/nm
(95.6% increase). Even response is-1.20475e-6 versus+1.80637e-4. Using the
old center changes that small even response to-7.29632e-7, without changing
the central derivative. The h0.5 peak3 spectral-Jacobian contraction gives
3.667761171813e-5/nm, differing by0.01364%; all peak indices are252.
Modal derivatives also depend on step:4.408675869771e-5/nm versus
2.144186733535e-5/nm. Two sizes do not establish convergence or its order.

CPU-only no-solve probe170921_0 completed in28seconds with all critical
evidence available. It compared723 mapped properties in each original
forward/adjoint layout: maximum discrepancy1.6941e-21m from te_point.
Saved Re/Im forward/adjoint coordinates match exactly. Generated seed mesh
and nominal index axes differ from saved coordinates by at most6.7763e-21m,
which is floating-point scale, not a meaningful displaced grid. Original FSP
hashes and companion HDF5 stats are unchanged; full HDF5 hashes were not taken.
Field shapes are inferred from coordinates, not full E/H extraction.

The observed adjoint Port2 shift-49.106nm is intentional: inspected vendor
port_fom.py adds sign*local resolution and a numerical-propagation phase.
Backward sign=-1 gives a1.25-step phase formula. Both saved phases are
23.2815894deg; the numerical wavevector needed to independently reconstruct
that phase was not exported, so formula agreement is not a fresh phase test.
These results do not support gross saved geometry/grid mismatch as the cause.
Next bounded question is step convergence and numerical dEps step dependence;
full calibration remains failed. See `s2_provenance_review.json` and
`s2_fdstep_two_sizes_comparison.json` under readiness evidence.

The coordinate API uses symmetry-unfolded getdata option2, checked against
[official Ansys documentation](https://optics.ansys.com/hc/en-us/articles/360034409834-getdata-Script-command).
No new field solve or old-project save was used for this provenance check.

## Interpretation plan for the pending four-step and dEps checks

Compare h = 2, 1, 0.5 and 0.25 nm without assuming smaller is automatically
better. Report successive derivative differences, signs, both nominal centers,
and peak3 spectral-Jacobian projections. A smooth halving trend on corr30 can
justify a chosen step for remaining class probes, but cannot establish the
full calibration. Nonmonotonic changes require further diagnosis; they do not
alone identify solver noise, a mesh defect or non-differentiable geometry.

The new stored-field probe must first reproduce its original h = 1 nm raw Re
contraction, then Im, before its smaller dEps steps are interpreted. It uses
original-forward Jacobians, whereas new forward differences use the repeated
center for even terms. Preserve that distinction. If the new extraction does
not reproduce the original contraction, diagnose the extraction/subset path
before treating a discrepancy as physics. No coefficient is fitted by this
probe. The six-class held-out gate and width/optimizer hardware gates remain.

## Additional numerical control to record, 08:08 Israel

Official [Ansys mesh-refinement guidance](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object)
recommends PVA for inverse design and distinguishes its `meshing refinement`
accuracy setting from the `mesh refinement` method. It documents default5
and maximum12. Our source sets the PVA method but does not explicitly set
that accuracy integer; the completed provenance probe did not extract it.
Therefore its actual value is currently unmeasured, not established as5.
The prepared no-solve dEps probe will record it without changing it. This is
a candidate control if finite-step sensitivity persists, not a diagnosed
cause or permission to substitute a different numerical family silently.

## New center evidence and next diagnostic, 06:43 Israel

The nominal repeat170901_0 completed exit0. Peak3(Tout) differs by+4.75118e-7
from the original; full Tout spectrum max difference1.22018e-6 and
RMS2.00109e-7. Wavelength/frequency grids match exactly and the parameter
vector/source identities were checked. This observed center difference is
only0.634% of the original corr30 +/-2nm central numerator; it is too small
to explain the large one-sided asymmetry by this measured center offset alone.
One repeat cannot bound run-to-run noise, eliminate all numerical effects or
validate adjoint-field provenance. The failed calibration is unchanged.

After explicit review, the exact new checkpoint SHA was recorded in the
remote center marker; serial array170906 now evaluates corr30+/-0.5nm.
The purpose is to measure step dependence and odd/even response, not to tune
one coefficient until a selected probe passes. Source files are frozen.
See TOP continuation and saved scalar/full-spectrum comparison for hashes,
resources, evidence paths and next actions. All original results are retained.

### Smaller-step interpretation prepared while the jobs run

For scalar f and step h, use odd O=(f+ - f-)/2, even
E=(f+ + f- - 2f0)/2, D=O/h and C=2E/h^2. Report even terms using both
old and repeated centers; the interval is center sensitivity, not a confidence
interval. At h2nm, corr30 O=3.74973e-5 and E=1.80637e-4; even response is
about4.82 times odd response. For a locally quadratic response, O shrinks
fourfold and E sixteenfold at h0.5nm. Deviations can also arise from smooth
higher derivatives; they do not alone diagnose a mesh kink or source error.

The conditional Richardson expression is [16D(0.5)-D(2)]/15. Two sizes do
not establish its assumed h^2 error law. Compare moving peak3 FD with both
the fixed center sample FD and the exact center peak3 Jacobian applied to
the spectral FD. Finite-step disagreement does not itself establish different
infinitesimal observables or a defective adjoint. Inspect the actual peak
indices, concavity and spectra before interpreting any discrepancy.

New exact Opus5.5 text-only review succeeded and is saved in
`readiness_evidence_20261007/claude_s2_center_repeat_review.json`. Its odd/even
and Richardson formulas were checked. Corrections: it wrongly imported the
shift30 wavelength slope into corr30; both stored corr30 endpoints have the
same sampled modal lambda1560.8782189638007nm as center. Its claimed0.045nm
corr30 peak displacement and derived0.18 fixed-wavelength change are unsupported.
Its noise-sigma expressions assume independent equal-variance noise, which
one repeat cannot estimate; no sigma/confidence bound is adopted. Shape-fit
residuals above the repeat difference or non-h^2 scaling do not prove mesh
error, since shape deformation and smooth higher-order response also contribute.
Do not prescribe another solve solely from those heuristic thresholds.

Prepared offline calculator `readiness_evidence_20261007/analyze_s2_fdstep.py`;
analytic quartic/Richardson, off-grid parabola and invalid-spectrum checks
passed exit0. It refuses absent endpoints and exclusively creates output.
After fetching exact task01/02 JSON and small NPZ spectra under the same local
evidence directory, run it with `--evidence-dir docs/readiness_evidence_20261007`,
`--old-spectra C:/Users/evyat/AppData/Local/Temp/s2_peak_failure_spectra_ppyzxc75/stored_port_spectra.zip`,
and a NEW `--output` path. This calculator is not a physics gate.

All twelve finite-difference legs and both peak-objective gradient
reconstructions completed. Fit job170754_2 failed intentionally after saving
its result because the scientific accuracy gate failed. The queue was empty
at the subsequent preflight. The read-only recovery watcher exited as designed;
this is not a VPN failure. Preserve the complete recovery chain and its sources.

## Independently verified verdict

The current peak3(abs(T_out)) fit has rank2, matching gradient signs, and
condition number8539.56. A sign match alone is insufficient.

| Probe | Fitted relative error | Leave-one-out error |
|---|---:|---:|
| corr1 | +16.70% | +37.74% |
| corr30 | +162.11% | +170.99% |
| avg1 | -8.31% | -10.34% |
| shift1 | +13.34% | +13.37% |
| shift30 | +0.64% | +315.81% |
| cavity width | -31.95% | -39.64% |

The unchanged gate requires every class and every held-out probe within10%.
All six held-out probes fail. The old softmax calibration also fails. Install
neither fit; no optimizer smoke or campaign may use these coefficients.

Independent NumPy reconstruction matches the saved coefficient and held-out
vectors. A separate linear program minimizes max_i|a*Re_i/FD_i+b*Im_i/FD_i-1|.
Its optimal worst relative residual is58.1015%, verified by direct substitution.
Changing least-squares weights cannot make one global complex factor satisfy
10% for these stored vectors. This identifies an inconsistency in the current
calibration evidence/model, not its physical cause. Do not introduce arbitrary
per-parameter correction factors to hide it.

Evidence: `readiness_evidence_20261007/s2_peak_fit_independent_review.json` and
the immutable `results_from_athena/recover_te_s2_20261006/watch/` checkpoints.

## Stored-spectrum checks, no additional FDTD solve

One read-only transfer preserved baseline plus all twelve small port spectra
in `C:/Users/evyat/AppData/Local/Temp/s2_peak_failure_spectra_ppyzxc75/`.
The archive SHA256 is
`b996142caf4a66d73336ed2845a564626e6fbd9e0e9a695ee89bf498e7522204`.
Its `analyze.py` reproduced every leg's recorded peak3 score within1e-12 and
verified identical wavelength grids. This rules out a local score-copy error.

The baseline T_out peak is at sample252 and the modal peak at250. This is a
real observable distinction, not proof of geometry mismatch. Grid spacing is
about0.004nm and recorded modal linewidth about0.2023nm (roughly50.6 samples).
The T_out parabola is concave, vertex offset-0.1442 samples, with valid weights
[0.0825151,0.9791982,-0.0617133]. A grossly unresolved few-sample peak is not
supported by this evidence. The original geometry/field provenance still needs
an actual check; the mean of shifted FD spectra is not a nominal-center solve.

One-sided peak differences relative to the stored center are highly asymmetric
for several probes: corr30 forward+1.09067e-4 versus backward-7.15698e-5 pernm.
That could reflect curvature, numerical effects or a baseline mismatch;
asymmetry alone does not prove the central derivative is inaccurate. The
shift30 endpoints move the peak by about45 samples each (~0.18nm). At this
finite step, applying the baseline Jacobian to the endpoint spectral difference
is not an infinitesimal derivative. Neither fixed-center nor baseline-weighted
endpoint diagnostics rescue the global fit. A smaller-step/provenance check is
needed before selecting a physical correction.

Evidence: `readiness_evidence_20261007/s2_peak_spectral_review.json`.

## Independent consultation, with corrections

The restricted ask-claude helper succeeded with response metadata confirming
exactly `claude-opus-5-5`. Its bounded review is saved as
`readiness_evidence_20261007/claude_s2_failed_fit_review.json`.
Adopted: inspect stored spectra/field provenance and finite-step behavior before
new optimization; do not install failed coefficients. Rejected or corrected:

- It said four held-out probes fail; the measured vector shows all six fail.
- It claimed a fitted corr1 sign flip; all fitted signs actually match. Raw
  Re/Im signs are not the sign of the corrected gradient.
- Raw quadrature magnitudes and FD/Im ratios do not prove failure of a complex
  linear combination. The independent minimax computation supplies that test.
- It called shift30 motion0.09nm; that number is a slope in nm/nm. Each2nm leg
  moves about0.18nm, comparable to this linewidth, not proven many linewidths.
- A one-sample peak offset or leg-average discrepancy does not prove fields
  belong to the wrong geometry. No such provenance verdict is established yet.
- The port-source/Jacobian separation is supported by inspected vendor code;
  changing `fom.fct` is read dynamically by `get_jacobian`, not a cached closure.

## Next bounded diagnostic and independent software progress

Prepare a distinct forward-only diagnostic at the exact recovered te_point:
first repeat the nominal center with the FD recorder/mesh settings, then review
baseline agreement. Only after that review consider corr30 ±0.5nm, preserving
the existing ±2nm data. No optimizer or adjoint is part of this diagnostic.
The center repeat answers a reproducibility question that the old center with
different recorder settings and perturbed endpoints cannot settle. Do not call
the diagnostic a new calibration pass; the full gate remains unchanged.

At05:47 preflight: Athena user queue empty, quota225G/300G soft (330G hard),
12h_4g available in QOS listing, shared license server/daemon up with0/50 solve
seats used. A100 diagnostic remains one job at a time,256G host memory,8CPUs,
the established compatible QOS. Recheck if dispatch is delayed materially.

The separately tested restart/modal candidate was integrated locally after
the chain ended. All thirteen integrated offline suites pass after repairing
one standalone test import path. No failed coefficients or opt-in seed flags
were installed; nothing was uploaded by that integration. Read
`TE_INTEGRATION_2026-10-07.md` for exact integrated hashes and retained logs.
