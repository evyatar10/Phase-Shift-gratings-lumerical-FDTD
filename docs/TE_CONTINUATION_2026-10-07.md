# S2 calibration and restart correction continuation

## Final local checkpoint 09:14 Israel - waiting for restored VPN

Both bounded SSH rechecks still timed out at09:11:47. No new remote state
is known. The existing30minute heartbeat was set PAUSED per AGENTS.md's
stop-on-genuine-blocker instruction; server jobs were not cancelled. Re-enable
this same heartbeat after VPN access returns, then reconcile existing jobs
before any dispatch. Do not create another schedule or claim unattended work
is still active while it is paused.

The local firstwave analyzer is integrated at
`readiness_evidence_20261007/firstwave_analysis/`; all six tests pass, with
no new FDTD evidence or measured report. Main SHA
3b18b5b0663ae2350cc4e3f1fc2c787b6a5d7b4fedaacfe38ba84b92febf384c;
repository-path pins SHA
cff7165a1cc6e7c4aae9024b6ce3dda8d1c9d62dfc007040bd0deaf6f14c7581.
README specifies exact original-byte checkpoint/NPZ fetch names and command.
It compares corr30 repeats and optional preserved shift30h2 data, verifies
frozen identity/grids/all saved scores, and never publishes scientific approval.
Remote success logs and retained FSP/H5 still require actual server inspection.
All Sol preparation workers finished; no pending worker or exclusive file owner.
The separate firstwave review contract preserves the unchanged10% global gates.


## Current access checkpoint 09:03 Israel - cluster visibility lost

IGUM SSH timed out at09:01:08 and Athena at09:01:39 (bounded10second
connections). Local Wi-Fi is up; the TAP-Windows VPN adapter is disconnected
and no132.68 IPv4 route was returned. This is consistent with lost VPN access,
not evidence of job failure. No cancellation, duplicate dispatch or VPN change.
Last verified: Athena170962_0-3 RUNNING08:57; IGUM101155_0 PENDING08:54.
Four new Athena forwards and one IGUM baseline remain uninspected;18 completed
new FDTD solves is still the measured count. Calibration remains FAILED.

Focused local Git checkpoint1d4cf6f preserves measured four-step analysis,
Re/Im corr30 checks, firstwave runner/markers/receipts, and skill snapshots.
A new no-solve five-class h1 harness is locally reviewed/prepared, NOT uploaded
or dispatched: `readiness_evidence_20261007/s2_other_classes_reviewed_0901.zip`.
Runner SHA0be8299d6607388f373876bf16b5f15e3d894220efcd770b9cccedd6f46d526d;
unchanged no-solve guard40dd487831da68080b29813cdbef2e1ce9e6629528dccd3d7c431dfc1229e99d.
Nine offline tests pass (run from repo root with PYTHONPATH pointing to repo),
and hidden local wcav mesher check passes without FDTD. Source targets only
indices0,60,120,149,295 at1nm and excludes already measured corr30. Serial
Re/Im128G,8CPUs,45min are a conservative prepared request, not a measured
runtime or scheduler acceptance. Re wrapper/preparation identity is recorded
in `s2_other_classes_preparation.json`; output is
`/work/results/s2_other_classes_h1_re_20261007T0901/other_classes_h1.json`.

After access returns: first reconcile exact170962/101155 queue/accounting and
immutable checkpoints (do not redispatch them), fetch/inspect firstwave ports,
then decide remaining4-11 class legs through a separate reviewed marker.
Before any new job, refresh resource/quota/shared seats including queued jobs.
For the new no-solve harness, verify frozen source hashes against firstwave
marker; upload only new exclusive helper/wrapper files, preserve old originals,
validate original checkpoint bindings, and dispatch Re once. Inspect its report
before serial Im. Full calibration, optimizer smoke and N98 hardware gates
remain open. No full optimization campaign is authorized.


User authorization: finish S2 calibration on Athena, correct/test campaign
restart persistence, then run a small hardware smoke after the calibration
gates pass. Full optimization is not authorized by this continuation.
The user also explicitly requested continuous progress through routine steps;
the project `AGENTS.md` records that instruction.

## Overnight extension and current continuation

## Latest measured checkpoint — both quadratures complete; first-wave 170962 running

**S2 calibration remains FAILED; full optimization and optimizer smoke remain
blocked.** Athena170926 tasks0–3 completed all four corr30 forward checks.
Eighteen NEW FDTD solves completed overnight. Stored-field CPU jobs170950_0
(Re) and170956_0 (Im) also completed, but performed no FDTD solves and are not
included in that count. IGUM101155_0 was still PENDING Resources in the latest
08:54 queue snapshot; do not treat this as a fresh live queue check.

The published role-aware four-step analysis reproduces scalar/spectral scores
and retains the historical h2 source qualifications: its center export hash
fingerprints extraction, not original FSP creation; the h2 endpoints have
separately pinned recovery-generated identities. Six analyzer tests and the
actual evidence analysis passed. corr30 peak3(Tout) central derivatives are:

| FD half-step nm | Derivative per nm |
|---:|---:|
| 2 | 1.874864880122e-5 |
| 1 | 3.695036628504e-5 |
| 0.5 | 3.667260923146e-5 |
| 0.25 | 3.491185204285e-5 |

All endpoint Tout peak indices are252. These measurements do not establish
an error order or a noise bound. Both nominal centers and spectral-Jacobian
projections remain in `readiness_evidence_20261007/s2_corr30_four_steps_summary.json`
and its full report; no Richardson estimate is promoted to a gradient gate.

Both no-solve probes reproduce their original h1 raw contractions: relative
differences4.108e-14 (Re) and5.248e-16 (Im). Raw quadratures vary less than0.1%
over dEps steps1,.5,.25nm. The old FAILED-fit weighted combination stays near
4.9142e-5/nm, within0.00171% of its h1 value, and remains uninstalled. This is
an unvalidated cancellation diagnostic, not a calibrated gradient. Original
FSP hashes/HDF5 stats are preserved. The Jacobian uses each ORIGINAL saved
forward, distinct from the repeated center used for new FD comparisons.
Evidence: `readiness_evidence_20261007/s2_deps_both_review.json`.
Both raw server reports measure PVA `meshing refinement`5 in generated,
original saved forward/adjoint and nominal-index contexts; no setting changed.
This replaces the earlier unmeasured-integer qualification, not the failed gate.

The lead's next bounded diagnosis is a new h0.5nm class check. First-wave
tasks0/1 REPEAT corr30 +/-0.5nm and2/3 measure shift30 +/-0.5nm. Existing
corr30 half-step data remain intact. Later tasks4–11 cover corr1,avg1,shift1
and cavity width ONLY after explicit first-wave review and a separate immutable
marker recording all four checkpoint SHAs plus the lead verdict. The live
index map is[0,29,60,120,149,295]; cavity is295, while180 is inert comb.
The new label is `lumopt2_te_s2_fdhalf_classcheck_20261007`, output root
`/work/results/diagnose_te_s2_fdhalf_classes/results/lumopt2_te_s2_fdhalf_classcheck_20261007`.

Runner preparation passed six offline test groups and actual hidden local CAD
task2: shift30+0.5nm,723 exact geometry properties and identical locked grids
after disabling optimization_dft. No local FDTD solve ran. Runner SHA
`4f5f9d21cb95f5cfbcb1e0fc91ae960dc0dd57449c973369051c551c07c7bba1`.
**Athena170962 array0–3%4 submitted08:55:24 Israel**, A100,8CPUs,256G host
memory,2h limit,12h_4g. At08:57 all four SLURM tasks were RUNNING: task0 on
n307 and tasks1–3 on n309; startup logs confirm8threads and GPU resource
readback. No first-wave result is claimed yet. Receipt
`readiness_evidence_20261007/s2_half_classes_first_dispatch.json`; verified
startup `readiness_evidence_20261007/s2_half_classes_startup_0857.json`.
First-wave immutable marker SHA
`6722e1883338c0259b6fd2aa7386a2d12f1c5124708f590c1ce30a7c266f8d02`.
Remote no-CAD identity validation passed. Only the new runner/marker/manifest
were written exclusively; existing production and IGUM sources were untouched.
At08:54 preflight: quota230G/300G soft/330G hard, shared seats0/50 occupied;
queued IGUM commits7 and this new array commits28, total35/50. These are
commitments, not measured running occupancy. Eighteen solves remain completed;
four new Athena forwards plus the queued IGUM baseline are outstanding.

Next monitor170962 at a reasonable cadence, then fetch exact first-wave
checkpoints and spectra after completion, inspect corr30 repeat consistency
and shift30 response, and
decide whether the separate later-stage marker is justified. Later tasks are
not approved merely because their source or offline tests exist. Preserve
frozen production and IGUM sources, current center SHA
`cd2dd4a58be7b83c46ef6a6ac90a874e5fc33edda884edfc8ac50ce5e3be4a06`,
and all original result history.


## Current update 08:33 Israel — four forward checks complete, Re validated

Athena170926 tasks0–3 all COMPLETED exit0 in31:04–32:24. Raw checkpoints and
ports fetched with preserved bytes into readiness evidence. Eighteen NEW
FDTD solves have now completed during the overnight work; IGUM baseline101155
remains queued (not counted). No full optimizer run occurred.

Re CPU probe170950 completed exit0 in5:18, MaxRSS19736488K. Original h1 raw
contraction reproduced to4.108e-14 relative; exact report SHA
73ae6c36fc75ac0c1a9ca470aff3c6627cb49c94f230b9b839810b5ade505440.
Original FSP hashes/HDF5 stats preserved. Re raw component varies less than
0.1% across dEps steps1,.5,.25. Generated and nominal server refinement5.
This validates Re extraction, not the combined/global calibration.

After explicit Re review, Im-only CPU probe **170956_0** submitted08:32:47,
same guarded script/128G/8CPUs/45min/noGPU. Output
`/work/results/s2_corr30_deps_im_20261007T0833/deps_sensitivity.json`.
Fresh shared pool08:32 was0/50 (finished forwards released seats), IGUM queued
A100 commits7. Check170956 actual startup/result next, then compare original
Im h1 and both quadratures' step sensitivity. No failed coefficients installed.

Four-step analyzer stopped before publication on a historical h2 source-map
comparison. Sol objective_code_audit is diagnosing that exact provenance
comparison in TEMP only; do not bypass it or duplicate the worker. New endpoint
identity/spectra checks passed up to that stage. Preliminary saved-scalar Tout
central derivatives h1=3.6950366285e-5, h.5=3.6672609231e-5,
h.25=3.4911852043e-5 per nm. These are not a completed global calibration or
proven convergence. Full validated analysis and Im interpretation remain open.
Latest local checkpoint **a3059fb** preserves prepared tools/tests/dispatch;
previous **f4fb0ee** preserves parallel rules/skills and convergence runner.

## Current update 08:21 Israel — no-solve derivative check dispatched

Athena **170950_0** submitted08:20:58, Re-only saved-field dEps-step probe.
No GPU/FDTD solve,8CPUs/128G/45min,2h_2g/l40s-public. Scheduler dry-run
170949 accepted; check actual startup next. Script SHA
f948aa8eb686f031624da5ef1d13a24457662fc90f6b44efd64686d7edd90cb8;
remote `/work/scripts/s2_deps_steps_20261007/probe_corr30_deps.py`, guarded
helper40dd4878...99d. New output
`/work/results/s2_corr30_deps_re_20261007T0820/deps_sensitivity.json`.
Uses ORIGINAL Re forward/adjoint pair, exact original-forward peak3 Jacobian,
three exact selected field slices, vendor assembly and single corr30 mesher
steps1,.5,.25nm. h1 must reproduce stored peak-gradient task00 before harness
fidelity is established; do not silently renormalize. Im not yet dispatched;
review Re first, then serial128G Im (240G QOS/user memory cap).
Eight offline tests pass; actual hidden local CAD exercises all three dEps
steps,4objects,shape1246x51x17x1, finite synthetic contractions and restored
nominal geometry. Local generated PVA refinement5; saved server value pending.
All frozen production hashes and reviewed provenance SHA checked before
exclusive new-file upload. Source package/hidden-CAD evidence preserved in
`s2_deps_probe_reviewed_0821.zip`; receipt `s2_deps_re_dispatch.json`.
Latest shared seats08:20 are28/50, plus7 committed queued IGUM. No new solve
seats for this CPU CAD/index check. Athena170926 four GPU checks remain active,
IGUM101155 remains pending Resources. Do not duplicate any of these jobs.

Four-step offline analyzer and six tests are prepared under readiness evidence
`four_step_analysis/`; six tests pass. Fetch names are documented there.
Original small port-spectra archive also preserved locally as
`s2_original_port_spectra_preserved.zip` (same bytes as original temp archive).
No real four-step report yet; new endpoint checkpoints are required. Both Sol
workers have completed their preparation; no worker owns production files.
Focused local checkpoint f4fb0ee saved parallel rules/skills/diagnostics; no push.

## Current state 08:03 Israel — four Athena diagnostics running

This section supersedes the historical status updates below. Athena array
**170926 tasks0–3** was submitted08:00:07 and all four were RUNNING at08:03,
tasks0–2 on n309, task3 on n310. Each has one A100,8CPUs,256G,2h under
12h_4g; startup logs confirm GPU and8threads. Tasks0/1 are corr30 +/-1nm;
2/3 are +/-0.25nm. Runner `runners/lumopt2_design/s2_corr30_convergence.py`
SHA12edb06b0f9cac1d7e1ff2bb3c36113a6c364007f8e7fe2c395fb49b1819a7bb.
Six offline tests and hidden CAD build passed; zero local FDTD solves.
The reviewed parent marker SHA is
312bd4e9e281b42eb53c38044627a20e4c37302338df6ab7c67691b1bfdec8c2.
Only new isolated files were uploaded; fingerprinted dependencies are frozen.
Output `/work/results/diagnose_te_s2_corr30_convergence/results/lumopt2_te_s2_corr30_stepconvergence_20261007`;
logs `jobs/logs/lum_array-170926_<task>.out`. Submission and startup receipts
are in `readiness_evidence_20261007/corr30_convergence_*.json`.

IGUM **101155_0** remains PENDING Resources at07:58: matching A100 baseline
qualification, not duplicate optimization. Shared pool was0/50 occupied before
Athena dispatch: four Athena solves commit28 seats plus7 for queued IGUM,
35/50 total. Athena home227G/300G soft quota. Recheck shared occupancy and both
queues before any further solve. Do not assume QOS limits reserve seats.

Completed: original failed calibration, nominal repeat170901, half-step
pair170906, and no-solve provenance170921. Fourteen new FDTD solves completed;
four Athena plus one IGUM are now additional outstanding solves. Do not count
scheduler dry runs or stored-field post-processing as new simulations.
Next: inspect/fetch the four endpoints and compare h2,h1,h0.5,h0.25 central
responses, even terms, exact peak3 Jacobian and spectra. One probe cannot
establish global calibration. No coefficients installed; no optimizer allowed.
A focused Sol worker is preparing a TEMP-only no-solve dEps-step sensitivity
probe; lead review/guard tests/resource checks required before any dispatch.
No worker owns production code. Keep this existing heartbeat, avoid duplicate
jobs, and continue independent preparation during waits.


07:54 update to the state below: Athena170921_0 COMPLETED exit0 in28seconds,
CPU-only/no FDTD. No meaningful saved geometry/grid mismatch was found:
all723 properties differ from te_point by at most1.7e-21m, saved field axes
are exactly equal, generated/index axes differ only at floating-point scale.
The adjoint port offset is intentional in inspected vendor code. Original
FSP hashes/HDF5 stats unchanged; see failure-report TOP and provenance review.
Proceed with reviewed bounded step-convergence preparation; do not repeat
the provenance probe or mistake it for a passed gradient calibration.

## Current state 07:52 Israel — both FD legs complete; two-cluster work authorized

The user explicitly authorized needed remaining gradient/diagnostic jobs on
both clusters in parallel and continual maintenance of relevant rules/skills.
AGENTS.md now records this; six updated skills passed their validators.
Installed-skill snapshots/hashes and a pre-edit backup are recorded in
`readiness_evidence_20261007/dual_cluster_skill_validation.json`.
Full optimization is still blocked by failed calibration.

Athena170906_1/2 both COMPLETED exit0,31m52s/31m41s. Checkpoint source/center
identities, exact grids and saved score reproduction passed the offline
comparison. All14 new overnight FDTD solves are now complete. corr30 peak3
derivative: h2nm=1.874864880122e-5/nm, h0.5nm=3.667260923146e-5/nm,
about95.60% larger. The even term changes from+1.80636948e-4 to-1.20475063e-6.
At h0.5 the exact center peak3 Jacobian applied to spectral FD gives
3.66776117e-5/nm (about0.01364% different from moving peak FD); all Tout peak
indices remain252. This establishes step dependence, not convergence or a
corrected global calibration. See `s2_fdstep_two_sizes_comparison.json`.

**IGUM101155_0**: baseline reproducibility qualification submitted07:41:30,
PENDING Resources at07:47; scheduler estimate15:46:55 is not guaranteed.
Accountacct-lumerical,part-preempt/qos-preempt,explicitgpu:A100:1,256G,8CPUs,
2h. No additional IGUM legs authorized by the runner's center gate yet.
Native part-preempt is a mixed A100/RTX PRO6000 pool; request actual GPU type.
Shared seats were7/50 at07:40 (Athena nowfinished; queued IGUM commits7).
All nine diagnostic source hashes match Athena.39/40 vendor Python modules
match local after line-ending normalization; only generated `_version.py` differs.
Original IGUM sources preserved in
`/home/evyatarrubin/research/bragg_sim_igum/source_history/before_te_parallel_readiness_20261007T0738.zip`,
341files,SHA503d9f6594b2dd61fc29d7938f6180ea6f17543111b6543aa65fb304fb0bde6e.
Updated IGUM via upload-only exit0, then explicit reference override pointing
to `data/te_s2_reference_20261007_original.json` (original SHA732ca575...13f).
Source fingerprints are frozen while this queued job consumes them.

**Athena170921_0**: no-solve field/geometry/grid provenance probe submitted
07:50:58, CPU-only8CPUs/64G,2h_2g,l40s-public,45min. First dry run on4h_0g
rejected64G before submission; verified that QOS caps memory at16G. The
2h_2g dry run accepted64G with zero GPUs. Check actual job/log next.
Probe `/work/scripts/probe_s2_provenance_20261007.py`, SHA
40dd487831da68080b29813cdbef2e1ce9e6629528dccd3d7c431dfc1229e99d;
output `/work/results/s2_provenance_probe_20261007T0753/provenance.json`.
Wrapper `jobs/run_s2_provenance_20261007.sh`, SHA
27f64ede5baa8fafa6a24e3d59a05ccb14178ca14a4d6a779482f7c37e2c518f.
Offline guards AND actual hidden local CAD constructor test passed, zero
local solves. Uses coordinate-only extraction, never saves old FSP, compares
original FSP hashes/HDF5 stats before/after. Generated artifacts are separate.
Field array shape is inferred from axes, not measured by transferring E/H.

New focused Sol audit found no six-column geometry Jacobian/unit error or
filename-map bug. It found a validation gap: vendor contraction checks sizes,
not coordinate equality. Probe tests that gap, not a proven coordinate defect.
Worker `/root/s2_gradient_provenance_audit` now owns TEMP-only preparation of
a four-task corr30 h1/h0.25nm convergence diagnostic, pending provenance review;
do not duplicate or dispatch it before parent review/tests. No production edits.
Next: inspect170921; compare its geometry/grid coordinates with proper symmetry
conventions. Check IGUM101155 only at reasonable cadence. Decide smallest next
step-convergence set after provenance; never install failed coefficients.

07:32 user steering: coordinate Athena and IGUM in parallel for suitable
independent authorized work, always checking the SHARED Lumerical seat pool.
Live read-only checks: Athena12h_4g MaxTRESPU GPU4/CPU512/mem1792G,
MaxSubmitPU100; this is that QOS's per-user limit, not a verified account-wide
four-job ceiling. IGUMqos-preempt user entry evyatarrubin shows MaxJobsPU16,
MaxTRESPU GPU16, MaxSubmitPU40; availability and other association/partition
limits still apply. qos-lumerical has no populated user limit entry in this
snapshot; that is not evidence of unlimited capacity. IGUM user queue empty.
Shared lmstat at07:32:7/50 seats in use, all seven held by our Athena n306
solve170906_2 (RUNNING18m28s). No new job submitted on either cluster.
Evidence `readiness_evidence_20261007/dual_cluster_limits_20261007.json`.
For future parallel diagnostics, count running AND queued commitments across
both clusters, GPU-specific seat demand and shared users; recheck before
release, preserve headroom and exclusive outputs. Verify IGUM source/wrapper
parity first. Current recovery/diagnostic reference paths are Athena-specific;
do not copy-dispatch them unchanged on IGUM. Calibration remains failed.

07:10 Israel heartbeat: SSH succeeded. Exact array170906_1 remains RUNNING
on n306 (28m32s);170906_2 is PENDING JobArrayTaskLimit as intended. Neither
task01 nor task02 checkpoint exists yet. No new wrapper error is visible;
the GPU-idle header is a launch-time snapshot, not current utilization.
Snapshot `readiness_evidence_20261007/s2_fdstep_status_0710.json`.
No duplicate dispatch or source change. Next scheduled check about07:40:
fetch any completed exact checkpoint/ports, but compute the central derivative
only after both endpoints pass identity checks. Current evidence checkpoint
is local commit `da5aa3d`; failed calibration and blocked optimization unchanged.

## Current state 06:43 Israel — center reviewed, smaller-step legs active

Nominal repeat170901_0 COMPLETED exit0 in31m50s. Same recovered parameter
vector and exact wavelength/frequency grids; all nine checkpoint source hashes
match local sources and were verified remotely before continuation. Preserved
FSP/HDF5/ports exist. New center peak3(Tout)=0.9257489820745417 versus
old0.925748506956398, delta+4.75118e-7. Whole Tout spectrum max absolute
difference1.22018e-6, RMS2.00109e-7. Modal peak delta-6.32945e-7;
sampled modal wavelength unchanged1560.8782189638007nm. Spectral FWHM
delta+2.56667e-7nm; this is not a spatial-width measurement.

Lead explicitly reviewed the scalar AND spectral evidence. Center peak delta
is0.634% of the old corr30 central numerator at +/-2nm. This supports the
planned smaller-step diagnosis; one repeat is not a statistical noise bound
and does not validate adjoint/field provenance or the failed calibration.
Exact NEW task00 SHA:
`cd2dd4a58be7b83c46ef6a6ac90a874e5fc33edda884edfc8ac50ce5e3be4a06`.
The matching `center_reviewed.sha256` was written remotely after this review.

**Array170906 tasks1/2, corr30+/-0.5nm, submitted06:41:57, serial throttle1.**
At06:42:59 task170906_1 (scheduler JobId170907) RUNNING n306; task170906_2
PENDING JobArrayTaskLimit. Both A100-public,12h_4g,256G,8CPUs,2h limit.
Early task1 log confirms8threads/build4572. No source upload: reused verified
sources, wrapper and existing manifest. Freeze fingerprinted sources through
these diagnostics. Preflight: user queue empty, quota226/300Gsoft/330Ghard,
shared licenses0/50, QOS mem1792G/GPU4 cap. Remote receipt:
`results/diagnose_te_s2_fdstep/results/lumopt2_te_s2_fdstep_diagnostic_20261007/legs_dispatch_20261007.json`.

Next: inspect exact task01/02 checkpoints and ports after completion, verify
their reviewed-center SHA, then compare central odd response and even curvature
at h0.5 versus h2nm, full spectra and peak3 scores. Do not refit a global factor
from one improved probe or call the calibration passed. No optimizer smoke,
N98 pilot or campaign is allowed while the existing10% gate fails.
Evidence in `readiness_evidence_20261007/s2_center_repeat_spectral_comparison.json`,
`s2_fdstep_status_0640.json`, and `s2_fdstep_legs_dispatch_verification.json`.

Actual usage query06:40 reports0% used (100% remaining) in its codex weekly
bucket, ordinary usage allowed, no secondary/model-specific bucket. This is
a change from6% remaining at05:49; cause unknown. No reset was redeemed by
this workflow. Preserve unknown model-specific/Claude limits.

Focused exact Opus5.5 review completed while the new legs run. Read the
corrections/protocol near the TOP of the failure report: Claude mistakenly
applied shift30's wavelength sensitivity to corr30. The prepared offline
`readiness_evidence_20261007/analyze_s2_fdstep.py` passed analytic checks;
use it only after fetching both endpoints. Failed scientific gates remain.
The existing30min heartbeat was updated and confirmed ACTIVE for170906.

06:10 Israel heartbeat: bounded read-only SSH succeeded. Exact job170901_0
remains RUNNING on n306, elapsed00:09:41; task00 checkpoint is absent and
the wrapper log shows no new failure. Saved snapshot:
`readiness_evidence_20261007/s2_fdstep_status_0610.json`. No new dispatch,
source changes or center-review marker. Recheck at the next scheduled
heartbeat (about06:40); inspect completed checkpoint and spectra before
authorizing either smaller-step leg. No additional audit is needed during
this wait; the failed calibration and blocked optimizer gates remain intact.

## Active diagnostic at 06:02 Israel — use this state first

Source fixes are committed locally as `1b2d75e`. The local integrated engine
and diagnostic were uploaded AFTER preserving remote sources and confirming
no jobs used them. **Freeze their fingerprinted sources while170901 is active.**
Old archive: `/home/evyatarrubin/bragg_sim_athena/source_history/before_s2_fd_diagnostic_20261007T055137.zip`,
348 source files, SHA`71d1e7e1f39a122c8d387b8628896b28dfc868f30771f2fb3388ded0b89fa9a0`.

New job **170901_0** RUNNING n306, A100-public,12h_4g,256G,8CPUs,
02:00:00 limit; exactly task0 was submitted (no ±legs). Start06:00:37.
Runner `runners.lumopt2_design.diagnose_te_s2_fdstep`, SHA
`3bf24e3ea47600f1c08f13803c7ab29bc6cf8b95c08fef63152168b1069ee361`;
remote engine SHA`53108bbd368dd699179ab65029628aa4c4b2d8ad687777f88b82f404a0952569`.
Both hashes were verified after upload. Early log confirms8threads/build4572.
Checkpoint expected under
`/home/evyatarrubin/bragg_sim_athena/results/diagnose_te_s2_fdstep/results/lumopt2_te_s2_fdstep_diagnostic_20261007/task_00.json`;
log `jobs/logs/lum_array-170901_0.out` relative to the Athena base.

Next: inspect accounting, exact task0 checkpoint and preserved ports; compare
new/old center spectra, not merely job exit. Only after explicit center review
may task1/2 be enabled by writing the exact NEW task00 SHA into
`center_reviewed.sha256` in that study's output directory. No ±task has yet
been submitted. The marker is not a calibration pass. Decide the next small
calibration diagnostic from measured reproducibility; preserve the failed10%
gradient gate and all old evidence. Do not restart the old recovery watcher;
the native heartbeat should monitor this new exact job directly at low cadence.

Six offline diagnostic checks passed. Hidden CAD actual task0 preparation
passed in `C:/Users/evyat/AppData/Local/Temp/s2_fdstep_hidden_build_nsak9n13/`:
hide=True enforced, zero solves, locked mesh unchanged with recorder disabled,
ports-58.018739/+58.067845um,501samples. An earlier probe's monkeypatch replaced
the lumapi class and failed before CAD; corrected test wrapped its constructor
method instead. No project code fix was required for that test-harness error.
Pinned reference hash is the ORIGINAL remote task00
`732ca5752ff2b0449a892ebafe7ba6a87a5ee1fce63bd929f8f7343bf50df13f`;
watcher's reformatted local JSON has different bytes but verified equal data.
The exact original is saved in readiness evidence. All13 integrated gates pass.
Last account allowance6% at05:49; no reset redeemed. Parent/worker preparation
is complete, no workers active. Calibration/optimizer hardware gates remain
FAILED/BLOCKED, not passed by this forward-only job.

## Current authoritative state after the final fit, 05:50 Israel

The recovery chain is terminal: all FD and both peak quadratures completed;
170754_2 FAILED the scientific gate after saving the fit. Queue empty at05:47.
Watcher42384 exited normally on the failed verdict; do not restart it for this
completed chain. Read `TE_S2_CALIBRATION_FAILURE_2026-10-07.md`: peak fit worst
162.11%, held-out worst315.81%; independent relative-minimax best58.10%.
No coefficients may be installed, and optimizer hardware smoke remains blocked.

The consolidated candidate is ALREADY integrated locally; **do not apply the
patch again**. All13 actual integrated offline suites pass. Current engine SHA
`53108bbd368dd699179ab65029628aa4c4b2d8ad687777f88b82f404a0952569`.
See `TE_INTEGRATION_2026-10-07.md`; defaults/S1/S2 configs unchanged. Remote
sources remain the original recovery version; no integration upload occurred.
The older frozen-source/candidate-pending paragraphs below are historical.

Stored-spectrum diagnostic reproduces all leg scores, shows ~50samples per
linewidth and strong one-sided asymmetry; no physical root cause proven.
Exact Opus5.5 review completed, with several claims corrected in the failure
report. A new forward-only nominal-repeat / corr30 step diagnostic is being
prepared by `/root/objective_code_audit`, exclusive ownership of
`diagnose_te_s2_fdstep.py`, its new gate and preparation doc. Do not duplicate
or edit those while active. Intended first dispatch is task0 ONLY after its
offline/build checks; ±0.5nm tasks require exact center-checkpoint review.
This is calibration diagnosis, not optimization. Preflight05:47queueempty,
quota225/300G,sharedseats0/50; full details and preserved spectral paths in
the failure report. Old jobs/results must remain intact; no cancellation or
cleanup authorized. Last measured model allowance8% at05:10, unknown now.

05:11 Israel milestone: all twelve FD legs completed exit0. Fresh bounded
read-only SSH confirms170752_0 (peak real quadrature) completed exit0,
170752_1 running on n306, and170754_2 dependency-pending. The real vector is
saved locally as `watch/recover_te_s2_peak/task_00.json`. Sources must remain
frozen through both remaining tasks. Watcher is alive, no access failure.

Independent local assembly saved the twelve FD checkpoint hashes and full
six-component derivatives for softmax, peak3(T_out), modal peak3 and sampled
resonance in `readiness_evidence_20261007/s2_complete_fd_vectors_0510.json`.
All twelve scalar outputs are finite and resonances inside the scan. The
legacy softmax least-squares diagnostic FAILS: corr30 residual+110.914%,
wcav-23.130%, shift1+10.404%; held-out errors also fail (corr30+114.045%,
shift30+101.991%). Matrix condition8736.66. These coefficients must not be
installed. This is not yet the CURRENT peak-objective fit verdict; wait for
its imaginary quadrature and independent fit review. Same-sign measured
modal/carrier FD derivatives here do not establish proportional objectives.
Account-wide weekly allowance at05:10 is8% remaining; ordinary usage allowed.

04:40 Israel heartbeat: watcher snapshot `20261007T012300Z` confirms
task170750_11 completed exit0 and170750_12 running on n306; eleven of twelve
FD legs complete. Peak reconstructions170752_0/1 and fit170754_2 are still
dependency-pending. Task11 scalar checks pass (finite, resonance in scan):
peak3(T_out)=0.9258512918524058, modal_peak3=0.9719206671388542,
lambda=1560.8822189724094nm. Watcher42384 is alive, stderr empty; no extra
SSH or source change. The fit runner saves its immutable result before
raising on a failed scientific gate: inspect its checkpoint even if the fit
job is FAILED, rather than interpreting that as missing evidence or rerunning.

04:10 Israel heartbeat: watcher snapshot `20261007T005256Z` confirms
task170750_10 completed exit0 and170750_11 running on n306; ten of twelve
FD legs complete. Task12 and the reconstruction/fit jobs are pending. The
new task10 checkpoint has finite scalar measurements and resonance inside
its configured scan. Watcher42384 is alive with empty stderr. No additional
SSH, dispatch, source edit or integration was needed. The completed N98
preparation and its six-test evidence are preserved in local commit811fa58;
final calibration, integrated gates and hardware validation remain pending.

03:40 Israel heartbeat: the watcher snapshot `20261007T002253Z` confirms
task170750_9 completed exit0 and170750_10 running on n306; nine of twelve
FD legs complete. Watcher42384 is alive with empty stderr, and no extra SSH
connection was needed. Task09 scalar checks pass (finite and inside scan):
peak3(T_out)=0.9254833733305935, modal_peak3=0.971286134143864,
lambda=1561.0582396498414nm. Tasks11/12, peak reconstructions and fit remain
pending. Prepared N98 packet/doc are being preserved in a focused local Git
checkpoint; this remains offline preparation, not hardware validation.

03:10 Israel heartbeat: bounded read-only SSH succeeded; task170750_9 is
still RUNNING on n306 (launched02:40), eight of twelve FD legs complete.
No failure or new immutable checkpoint appeared; tasks10-12 and final
reconstruction/fit remain pending as expected. Watcher42384 is alive with
empty stderr. Account-wide weekly allowance is 9% remaining, ordinary usage
allowed; conserve it for final-fit review, integration and hardware evidence.
No new dispatch, source change or repeated audit was warranted at this check.

N98 preparation completed after the 02:40 check: read
`TE_N98_PILOT_PREPARATION_2026-10-07.md`. Six focused offline tests passed;
the exact original S2 seed/bounds and pilot options are preserved in its
JSON/archive. It is a blocked preparation helper, not a launchable study or
hardware pass. The worker has released ownership. Calibrated perturbed
`te_point` differs from the original seed; directional qualification at the
actual starting geometry remains necessary. For measured width comparisons,
use the total v3 moving-width prediction, not `dw_pred` alone. Archive SHA256
`b35cf728d62f462a5da29327d48c10a0b06c3131830c4e30073fb7e13fe5f979` was
independently checked. The 02:40 status checkpoint is committed as d23e6ac;
no server/source changes accompanied this preparation.

02:40 Israel heartbeat: task170750_8 completed exit0 and task170750_9 is
running on n306; eight of twelve FD legs complete. The new task08 scalar
checkpoint is saved, with peak3(T_out)=0.9258425299757719,
modal_peak3=0.9719109009213397 and lambda=1560.8782189638007nm. Final
reconstruction/fit jobs remain dependency-pending. Watcher42384 is alive;
one bounded read-only SSH check was used to refresh the older snapshot.
No integration or dispatch occurred. The existing Sol worker
`/root/objective_code_audit` now owns a NEW isolated fail-closed N98 pilot
preparation and `TE_N98_PILOT_PREPARATION_2026-10-07.md`; do not duplicate
that work or modify its files while active. Scope is offline runner/launch
packet preparation only, no CAD/server action, calibration hash unset and
qualified field-gradient evidence must remain visible. Prior candidates and
production remain untouched. Inspect the worker's handoff before integration.

02:10 Israel heartbeat: one bounded read-only SSH check succeeded. Task
170750_7 completed with exit0 and task170750_8 is running on n306; seven of
twelve FD legs are complete. Tasks9-12 are queued at the serial throttle,
170752_0/1 and170754_2 remain dependency-pending. New immutable task07 was
saved by the existing watcher helper. All seven completed legs have finite
scalar observables and resonance inside their configured scan; this is not
the pending calibration fit or gradient-validation verdict. Task07 has
peak3(T_out)=0.9258033986058473, modal_peak3=0.971867567526548,
lambda=1560.8782189638007nm. Quota221G/300G soft, watcher42384 alive with
empty stderr. Ordinary Codex usage is allowed with 10% weekly allowance
remaining. No new job, upload, cleanup, or production edit occurred.

At 01:43 the user reaffirmed overnight continuation toward morning TE
readiness. The existing heartbeat was updated, not duplicated, to include
the morning evidence handoff and the user's periodic local-commit permission.
The read-only watcher is alive with no stderr; calibration still has six of
twelve FD legs complete. Full-campaign dispatch remains outside this scope.
Morning handoff must distinguish completed checks from pending/failed gates,
and include tested code identity, calibration verdict, objective/adjoint
decisions, actual hardware evidence and exact launch/resume commands when
available. Do not convert the morning target into an unsupported readiness claim.

Heartbeat at 01:40 Israel time found no change since the fresh 01:37 watcher
snapshot. Account-wide weekly Codex allowance is 11% remaining; ordinary
usage is still allowed. Preserve capacity for the final calibration decision
and integration, rather than repeating completed audits. No Git mutation,
server connection or new dispatch was made in this follow-up. The next
scientific action remains inspection of the full peak3 calibration vectors
and unchanged held-out gates after the dependency chain completes.

Readiness status at 01:39 Israel time: the fresh read-only watcher snapshot
`20261006T223738Z` has task170750_6 complete and task170750_7 running;
six of twelve FD legs are complete. Peak reconstruction and final fitting
remain pending. The watcher process42384 is alive and its stderr is empty.
The small hardware smoke and N98 validation have not run. Thirteen offline
suites passing is not full optimization readiness. This check used the fresh
local snapshot and did not create an additional SSH connection or submit jobs.

Latest formulation review: read `TE_OBJECTIVE_DECISION_2026-10-07.md` for the
user's transmission/constant-width priority, wavelength freedom, the corrected
AL formulation, actual solve count and the reuse recommendation. Claude Opus
provided a substantive review, then its claims were checked/corrected against
live code and numerical algebra tests. The new opt-in measured-modal guard
passed all 13 offline suites, including ten focused driver regressions. Read
`TE_MODAL_GUARD_CANDIDATE_2026-10-07.md` and
`te_modal_guard_candidate_location.json` for the preferred isolated candidate.
Its engine SHA is `d11f34db9777dd1d9deba6ab19372238c046e8d705a5c9c8c4281702e88b9559`.
**Apply only `te_modal_guard_candidate_20261007.patch`, not both patches.**
It consolidates the previous fixes; the earlier candidate/archive remain
preserved as historical evidence. The new archive is
`readiness_evidence_20261007/modal_guard_candidate_20261007_52ed8b9e.zip`
(SHA256 `82ea60faa70780021183be2b8c8c04296b4baca48585dcee34eda45e84b83187`).
The production engine remains frozen; integration and hardware validation are
pending. After dependencies finish, recheck the manifest, integrate this one
patch, and rerun all thirteen gates from the actual integrated tree. Pilot
options are `wgp_modal_guard=True`, `best_metric="modal_peak3"`; defaults stay
off and existing S1/S2 configurations are not activated by this record.

Read-only Athena access at 01:30 succeeded: five FD legs complete,
170750_6 RUNNING on n306, remaining chain pending, quota234G/300G soft.
The task6 wrapper confirms eight threads and build4572. Its launch-time GPU
header is not a current GPU-utilization measurement. No submission, source
upload, cancellation or cleanup was performed by this check.

At 01:33 the account-wide weekly allowance was12% remaining (same reset,
ordinary usage allowed). Conserve model capacity for calibration decisions,
integration and hardware evidence. Use brief status-only follow-ups while
the unchanged serial dependency chain is running; do not repeat these audits.

The user subsequently requested autonomous overnight readiness work, Sol
workers, and a tested Claude fallback, with no routine questions. Full
optimization is still not authorized. The completion criteria are now explicit
in `TE_READINESS_GATES_2026-10-07.md`; preserve all failed scientific gates.

At 01:07 Israel time, the watcher confirmed task170750_5 completed and
170750_6 running: **five of twelve FD legs complete**. Remaining FD tasks,
170752_0/1 peak reconstruction and170754_2 fit are still pending. Watcher
PID42384 is alive; its stdout is change-only, so an old printed timestamp is
not proof polling stopped. No new job was dispatched during this overnight
audit. Production source hashes remain frozen.

Native thread heartbeat `s2-overnight-inverse-design-readiness` is ACTIVE every
30 minutes for this thread. Its prompt reads these records, avoids duplicate
work and now includes budget-aware routing. It is separate from the read-only
Athena watcher. Host connectivity and account allowance limit future model
execution; neither an automation nor a skill guarantees unlimited operation.

Independent Sol reports:

- `TE_ALGORITHM_AUDIT_2026-10-07.md`: confirmed acceptance-reference poisoning,
  stale-width lookup, repeating missing-width trials, missing-gLambda step
  protection, stricter-band selection trap and global evaluation-budget gap.
- `TE_SAVED_RESULTS_REVIEW_2026-10-07.md`: stored S1 modal T0.95276031 at
  W19.385006µm is +1.38% wider than its anchor. Its original bands allow that;
  a literal no-growth claim does not. Missing final raw profiles and the
  recorded proxy residuals remain evidence limits.

Candidate fixes for accepted-only FOM reference, newest-row measurement,
bounded missing-width retries and fail-closed v3 wavelength-gradient handling
passed v3, projection and12 persistence tests at engine SHA
`a8ec9be98ee49ba0651526bd0c551f87b7dd4c5e20140e535177041104e42cb8`.
That intermediate hash is superseded by the completed candidate below.
The shared hard-width option and persisted global forward-evaluation budget
are now integrated **in the isolated candidate only**. All twelve offline
suites pass (exit0), including fourteen persistence tests, fourteen budget
helper tests, six real-campaign budget tests, six smoke guards and strict
width acceptance/selection regressions. Defaults and existing S1/S2 seed
configurations remain unchanged. Budget units are geometry forward batches,
not individual engine jobs or adjoints. Interrupted reservations stay spent.

Previously tested 12-suite candidate engine SHA256 (preserved, superseded by
the consolidated 13-suite candidate above):
`2558954f23df132a57f6de259e2a60ee43bf62012ce87617913b288117a38587`.
Regenerated patch SHA256:
`d4cf42eec8844e4a87c2d3dbe7d321ff1d608bdbbde0b795f1206286469dc14e`.
`git apply --check` passed without applying it. The expanded location manifest
records six candidate files and seven root dependencies with hashes; the new
campaign-budget gate is included in the patch. Full commands, exit codes,
hashes and logs are in
`readiness_evidence_20261007/budget_integration_gate_results.json`.
Recheck manifest hashes, preserve the frozen baseline, and apply/integrate
only after every dependent recovery task has finished. Rerun the actual
aggregate from the integrated tree before upload. No source was uploaded and
no additional solve was started during this candidate work.

The complete tested candidate, shared dependencies, patch, manifest and gate
record are also preserved byte-for-byte in
`readiness_evidence_20261007/tested_candidate_0118.zip` (16 files, SHA256
`a550be9f508c08d15e27a4185a9246a008b909fe337f9df231f551e68f73f78a`).
This removes reliance on the temporary candidate directory for recovery.

The prepared smoke now fingerprints both new helpers and its own runner,
and checks the persisted budget as well as evaluation/profile/state evidence.
Its calibration hash is still unset, deliberately blocking CAD/solves until
the actual peak3 fit passes review. Existing hidden-CAD generation established
geometry/plumbing only; all subsequent edits were optimizer/persistence logic.

The installed `budget-aware-research` and `ask-claude` skills are linked from
AGENTS.md. Ten usage-summary tests and a live exact `claude-opus-5-5`
text-only consultation passed. Claude's review is advisory, not a physics
gate. `BUDGET_WORKFLOW_2026-10-07.md` records the exact evidence. The account
reported18% weekly Codex allowance remaining, reset October12 at10:44 Israel
time; no Astra-specific allowance or Claude remaining allowance is known.
No reset credit was redeemed. The sanitized actual snapshot is
`readiness_evidence_20261007/usage_snapshot_0106.json`.
At the 01:19 stage-boundary check the same account bucket had **15% remaining**,
with the same October12 reset; ordinary usage was still allowed and the reset
credit remained unused. Do not attribute the account-wide change exclusively
to this task. The watcher PID was alive with empty stderr, and the native
automation view plus saved configuration confirmed the existing ACTIVE
30-minute heartbeat. Candidate work is now checkpointed; avoid new broad
audits while waiting for the calibration evidence.

Latest user steering: prioritize Claude reasoning/reviews now because Codex
capacity is scarce, while expecting intermittent Claude failures. The global
rules and budget-aware skill record this priority. A failure must be visible,
without repeated immediate probes; continue independent work or a bounded
Sol fallback where needed. Claude availability does not gate unrelated work.

VPN note from the user: access may intermittently become unavailable and
require reconnection; this was a future contingency, not a reported current
outage. The cluster skill now records bounded connection attempts, preserving
jobs/last known state, and reconciliation after access returns. The current
watcher exits on an SSH error, so check its error log and process liveness
before restarting one copy after connectivity is restored. Never infer job
failure or resubmit solely because the VPN is disconnected.

## Calibration evidence at 00:37 Israel time

170746 tasks 1, 2, 13 and 14 completed. Both stored-field gradient
reconstructions match the original printed gradients. 170750 tasks 3 and 4
completed; task 5 is running. Four of twelve FD legs are available. Tasks
6–12, peak3 reconstruction 170752_0/1 and fit 170754_2 remain queued behind
the serial throttle/dependencies. No final calibration verdict exists yet.
Quota is 237G/300G soft (330G hard). Scalar evidence is preserved under
`results_from_athena/recover_te_s2_20261006/watch/`.

The frozen shared sources remain unchanged. Do not deploy the candidate
below while those jobs depend on their current source fingerprints. Refer to
`TE_RECOVERY_2026-10-06.md` for the job chain, paths and resource checks.

## Isolated restart candidate

Candidate location is recorded in `te_restart_candidate_location.json`;
`te_restart_candidate_20261007.patch` carries the changes to existing source
files. New shared helper: `runners/lumopt2_design/persistence.py`.
New failure-injection gate: `runners/lumopt2_design/gates/gate_persistence.py`.
The production engine is not yet wired to that helper.

Changes in the candidate:

- Immutable evaluation/profile IDs, so repeated eval0000 cannot replace an
  earlier profile; MAC uses each row's recorded profile path. A missing
  required profile rejects the trial, even if a stale legacy file exists.
- Recoverable unterminated JSONL tail: readers preserve the original; the
  next writer archives it before atomically publishing the repaired log.
  Corruption in a completed record is fatal. All driver append paths use
  this behavior, including the missing-width retry branch.
- Missing optstate means fresh state; corrupt or invalid state fails with
  an explicit recovery message. Atomic writes retain previous/new versions.
- Committed acceptance IDs prevent a logged but unclassified trial from
  becoming the restart best after an interruption. Legacy rows remain
  readable under their old eligibility rules; historical missing decisions
  cannot be reconstructed retrospectively.
- Gradient artifact names use UUIDs, avoiding same-second restart collisions.
- The final reported best FOM comes from the same eligible row as its
  parameter vector; previously it could report the last accepted FOM beside
  a better historical vector. A real `run_campaign` regression test covers it.

The original candidate passed all seven existing offline suites. Committed
acceptance IDs passed the complete nine-suite run (including the new
persistence and smoke guard suites). After fixing the final best-FOM report,
eleven persistence tests pass, including real callback/driver tests and
deliberate publication failures. Rerun the aggregate after integration.
Logs live beside the candidate as `*.candidate.log`. These are offline tests,
not evidence of a new FDTD solve or a completed hardware smoke.

Persistence assumes one writer per campaign directory. State/profile history
is intentionally retained. Do not silently recover a corrupt active state
from an arbitrary history file: inspect matching evidence first.

Original frozen sources have been preserved and byte-verified in
`readiness_evidence_20261007/frozen_recovery_sources_20261006.zip`
(SHA256 `339c182a464eb8830232359527af4a06a3d6cc493c980b88e5fa9dc496992cc6`).
The engine remains `8f09350583a53146091faf590c2c93e8d93ac01a46b48006890db62c72a295e5`.

## Prepared hardware smoke

`runners/lumopt2_design/smoke_te_s2_restart.py` derives the existing task38
pipeline surrogate: N70/free60, 491.06nm pitch, 350nm height, TE bare,
1560.464nm center, 2nm/501-point scan. Two evaluations per invocation and no
automatic recenter retries. This is explicitly not N98 physics validation.
It refuses CAD/solves until an exact reviewed calibration-file hash is filled
in, rechecks the actual peak3 vectors/10% held-out gate, and audits immutable
profiles, committed state, MAC, v3 step and FOM=T_out peak3 before completion.
Five offline smoke guards passed. No server dispatch has occurred.

Hidden local CAD generated this N70 project successfully with provisional
(1,0) port coefficients for build testing only. Monitor assertions passed:
ports at +/-44.318um outside grating +/-34.497um, TE symmetry and frequency
sampling correct, single-frequency twin explicitly pinned. No solve ran.
Artifacts: `C:/Users/evyat/AppData/Local/Temp/s2_restart_smoke_build_szfccjdw/`;
`build_only_pass.json` records completion. The PowerShell `*>` wrapper returned
1 on benign Python INFO stderr; a no-CAD probe reproduced that wrapper error
with an actual Python exit0. Use subprocess capture for future exit reporting.

## Next actions

1. Keep the existing read-only watcher active; inspect each new scalar result.
2. Review both final calibration fits, including peak3 per-class and
   leave-one-out errors at the existing 10% gate. Do not relax or transplant
   coefficients. Save full precision and provenance when installing a fit.
3. Preserve a snapshot of the original frozen sources; integrate the tested
   restart candidate only after dependent recovery tasks finish.
4. Run the bounded hardware smoke with explicit calibration and fresh output
   identity; inspect width, wavelength, MAC, step and persistence evidence.
   Calibration failure is a scientific blocker to that smoke, not a reason
   to launch a full campaign or substitute S1 coefficients.
