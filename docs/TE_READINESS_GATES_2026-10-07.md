# Full-optimization readiness criteria

## Update 14:13 Israel — evidence map and runtime-aware plan

The user's requirement to independently qualify physical gradients and plan
expensive simulations ahead is now in AGENTS.md, work-alone, cluster-ops and
inverse-design skills. All three changed skills passed validation; before/after
snapshots are preserved. No production solver source changed.

Read TE_VALIDATION_EVIDENCE_MAP_2026-10-07.md for original Claude claims versus
independent checks and gaps, and TE_VALIDATION_PLAN_2026-10-07.md for the
conditional experiment sequence, measured runtime/resource estimates and
prepared post-processing. Historical S1 held-out port error -16.5% and S2 field
shift1 residual +23.8% remain explicit limitations; neither is a blanket pass.

At14:10 Athena171236 tasks3-5 remained RUNNING at81-83 ps/150 ps, about23 minutes
engine time remaining. Estimated outputs14:33-14:40, not guaranteed. Count29
completed FDTD solves remains unchanged. Exact150ps strict fetch and analysis
are still next. Usage14:10 reports58% weekly allowance remaining; model-specific
and Claude allowance are unknown. Full readiness remains NO.

## Update 13:55 Israel — independent spectral analysis completed; 150 ps still running

The saved 100 ps comparison was independently reproduced. Five focused
arithmetic tests passed; the parent reran the actual analysis and obtained
byte-identical numerical output. All source/input hashes and the plotted
figure were checked. The report and figure are in readiness_evidence_20261007:
`s2_temporal100_width_normalization_report.md` and
`s2_temporal100_spectra_comparison.png`; full reproducible packet1356 retained.

Measured nominal spectral FWHM changed from0.209867 to0.209421 nm for Tout,
and from0.202344 to0.201610 nm for modal transmission. These are spectral
linewidths, not the spatial-width design constraint. At fixed bin252,
modal transmission increased0.00811158: the exact endpoint ratio decomposition
has outgoing numerator term+0.00786251 and incoming-normalization term+0.000249071.
Both observables and the numerator changed; normalization alone does not explain
that comparison. This is algebra on measured endpoints, not causal attribution.
Raw sourcepower is absent from these NPZs and was not inferred from port P.

A separate sampled modal balance check gives nominal1-|S21|²-|S11|² of
0.0259228 earlier versus0.0183103 at100 ps. This residual is not an independently
measured radiation/absorption loss and does not certify energy or convergence.
The packet uses R for Tout/modal ratio; the separate balance report uses R for
reflection. Keep those meanings separate. All failed calibration gates remain.

At13:53 all171236 tasks3–5 were RUNNING, approximately33.07 ps/150 ps,
with no completed checkpoints and approximately40 minutes of solver time
remaining. Actual startup logs verify eight threads and separate physical
GPUs, including two different PCI bus IDs for jobs sharing n311. The cluster
skill now explains job-local GPU0 numbering; it passed skill validation.
No new jobs or production-source changes were made during this follow-up.
Next dependent scientific step remains the exact150 ps completion/fetch/analysis
workflow below. Completed new FDTD count29; latest committed evidence0ded829.

## Update13:41 Israel - 100ps completed and materially changed spectra;150ps171236 released

All171216 tasks0..2 COMPLETED exit0 in35:45-35:51. Exact100ps/status1,
1499020steps, dt6.671026479768415e-17s; frozen controls/source/API/grids and
remote FSP/H5/NPZ/log content hashes passed cold verification. Strict local
reader passed actual inputs. Completed FDTD count now29. Exact pinmap SHA
94b062900bd3765dd4a3ef78543dd6998378720ed429264e5aa7833339d5e919.
Raw100ps CP/NPZ files retained; review s2_temporal100_review_1339.json and
full501 spectra s2_temporal100_spectra_1339.json, compact summary1341.

For fixed h=.0625nm, Tout peak3 derivative changed -.001763296→-.000208331
pernm, modal -.002032697→-.000316571 pernm. CenterTout increased.007537546
to.933286528, modal increased.007954718 to.979764300. Same-duration even terms
fell to+6.47534e-6 Tout,-1.58083e-6 modal. This measures substantial temporal
sensitivity, not convergence/noise bounds or restored calibration. Old adjoint
fields remain tied to old stopping conditions. No old S1 result is disproved.

Lead reviewed full100ps results and released ONLY150ps tasks3..5 with separate
immutable marker /work/data/s2_temporal150_review_20261007_1257.json SHA
6cc6d9fd8cab5419b436b433725fbb2cc88b49f538628f897929217940527f51.
New Athena array **171236_3..5** submitted13:40:15,3A100,8CPU256Geach,2hlimit,
same frozen555f runner, no source upload. Output same temporal family.
Fresh13:39 queue empty,226G/300Gsoft,shared0/50used plusIGUM101155queued7;
newAthena21 plusqueuedIGUM7=28/50. IGUM stillpendingResources.

Next inspect exact171236 status/checkpoints/logs; then use prepared strict150
fetch helper C:/Users/evyat/AppData/Local/Temp/s2_temporal150_fetch_zpn4skx9/
fetch_temporal150.py (six tests parentrerunpassed), explicitjob171236, markerpath
above, exactSHA, existing s2_temporal_env_1257.json, NEW absolute output-dir.
Archive s2_temporal150_fetch_reviewed_1333.zip. Reader supports all5optional150
arguments; use both independentlypinned100/150maps. Compare fulltriplets at
same h, entire spectra and same-duration Jacobian projections. Do not run any
new solve by calling main during retrieval. No optimizer/coefficient release.
At13:42 all171236tasks RUNNING on n310/n311 and actual GPU solver initialization
passed; notcompleted. Usage13:41weekly62percentremaining, separatepoolsunknown.
Latest localcommit498a126 before this new evidence. Current fullreadinessNO.
Clarification to user: original code DID have a physics-related implementation
bug: wavelength-selector adjoint differentiated Tout while its spectral slope
used modal S power (~5percent mismatch on saved S1/S2). Fixed/regressiontested
as documented in TE_RECOVERY_2026-10-06.md; full optimizerhardware stillpending.
No demonstratedwrong geometry/TE monitorplacement. OriginalS1scalar evidence
remains preliminary, not disproved by newS2temporal tests.

## Update13:27 Israel - both raw quadratures reproduced; fixed-bin comparison remains diagnostic

CPU171225_0 Im COMPLETED exit0 in8:24. Raw original SHA
1470c5695e08b7d939d2c4fbfa63bbbe0d6a0cdd2ea7cd36a48544ff397a9d7c.
Originalpeak3 -1.7084692721685926 matches reference -1.708469272168592
(relative3.90e-16); all originals/dependencies preserved. Three raw Im bins
244/252/260: -1.4055662051364066,-1.7095314148810021,-1.4984896535677845.
With reviewed Re, three-bin/two-coefficient matrix condition4.3204. Diagnostic
LS relative residuals h=.125/.0625: Tout0.1630%/0.1132%, modal4.4911%/4.8925%.
However heldout center errors Tout -0.00112158/+0.000788831 pernm and modal
+0.0349122/+0.0385469 pernm remain; large flanks dominate norm. One parameter,
three bins, variable auto-stop times: no global calibration or peakgradient pass.
No coefficients installed. Exact comparison/LOO evidence in
s2_fixed_bin_comparison_1328.json and compact s2_fixed_bin_summary_1328.json.

At13:23 temporal171216 allthree RUNNING, actual62-63ps/100ps. CompletedFDTD26.
100ps retrieval pending;150ps blocked until explicit100ps review. Parent reran
all7 strict offline reader tests exit0; packet s2_temporal_reader_reviewed_1323.zip.
Latest actual usage13:25 weekly64%remaining; model-specific/Claude limitsunknown.
Personalinverse-design skill updated/validated with measured arithmetic-only
reload guard and one-parameter fixed-bin limitations. Focused Claude Opus5.5 review completed; accepted one-residual-degree-of-freedom
warning, rejected tautological cross-step "FD-limited" criterion and150ps center-only
interpretation. Review claude_fixed_bin_review_1327.json. Latest Gitdf279ba before this checkpoint.

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

## Update12:35 Israel - CPU guard failure preserved; Im remains blocked

Athena171195_0 ended FAILED exit2 at12:29:49, elapsed8:59,MaxRSS~19.0GiB.
The original shift30 peak3 control reproduced0.20875441140950296 versus
0.20875441140949622 (relative3.23e-14); original files and production code
identities remained unchanged. The first one-hot attempt stopped with
`RuntimeError: Vendor normalization changed across objective Jacobians`.
No fixed-bin contraction completed and Im must NOT be dispatched yet.
Raw report s2_fixed_bin_re_original.json SHA
 d6ab05927276d43ceea194b23f1cccbd389abd586b3ee0dccae2fe71efb6b454.
Sol objective_code_audit owns isolated diagnosis/instrumentation; do not relax
exact normalization guard without measuring the offending metadata change.
Existing deployed helper/output stay frozen. No new FDTD solves.

Temporal runner remains TEMP-only preparation. Hidden CAD accepts explicit
FDTD `use early shutoff=0` with100/150ps while other settings stay unchanged;
actual termination has not been tested. Claude's verified Opus5.5 review is
saved with rejected causal/noise/tolerance claims in claude_temporal_review_1234.json.
No arbitrary proposed tolerance became a readiness gate. The maintained
inverse-design skill now covers actual termination checks and the builder's
falsey-zero pitfall; skill validation passed. Git checkpoint db329fa preserved
26solve evidence before this new CPU failure. Calibration remains FAILED.

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

## Current status 11:18 Israel — diagnostic progress, no calibration pass

Access restored at11:12; all four170962 tasks completed exit0 around09:27.
The first-wave analyzer and remote cold-resume/artifact checks passed their
narrow evidence gates. Twenty-two NEW FDTD solves are complete; global S2
calibration remains FAILED, and optimizer smoke/full optimization remain
blocked. The existing heartbeat resumed ACTIVE at11:18.

corr30 h0.5nm repeats differ by+0.75593% in Tout derivative and-5.23261%
in modal derivative. shift30 h0.5nm derivative is-2.7032781e-4/nm, while its
fixed-center finite-step spectral-Jacobian projection is+3.8585713e-4/nm
with11-bin peak motion. These finite-step measurements establish neither
an infinitesimal adjoint defect nor a universal small-step family rule.
**Remaining FD tasks4–11 are held; do not release their marker yet.**

CPU-only Re171145_0 was submitted11:17:22 for original h1 scalar contraction
checks of five other classes (128G,8CPUs,45min,zeroGPU/noFDTD). Inspect its
results before serial Im. The lead has not adopted Claude's unsupported
quadratic peak-ripple claim; the bounded TEMP-only current-objective test
is pending. No full wavelength-resolved adjoint Jacobian is currently exported.
The unchanged10% per-class/held-out gate is not replaced by these checks.

IGUM101155_0 remains queued Resources at11:15; its11:12 accounting DB error
was not an SSH failure. Preflight pool0/50 occupied, queued IGUM commits7,
quota233G/300G soft/330G hard. Re CPU processing adds no FDTD solve seats.
See current continuation and readiness evidence
`s2_firstwave_measured_1113.json`, `s2_other_classes_re_dispatch_1118.json`.

## Historical access blocker, 09:03 Israel

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
