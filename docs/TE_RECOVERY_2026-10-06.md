# TE calibration recovery — 2026-10-06

The user explicitly requested S2 calibration, gradient-job recovery and the
FOM/transmission investigation, and selected Athena. This supersedes the earlier
stop state **for this recovery only**. TE, N=98 per side, 60 free periods per side,
no posts and the existing near-20 µm width bands are retained. No campaign has
been restarted. S2's `ADJ_FIX_PORT` remains unset pending a measured passing fit.

## Verified FOM discrepancy and a related bug

Athena extraction job **170745_0** completed successfully (1m21s, ~9.24 GiB peak
RSS). It read surviving solved files and did not run FDTD. Small port spectra
and metadata are preserved in `results_from_athena/recover_te_s2_20261006/`.

| Stored forward | modal peak3 | T_out peak3 | modal sampled peak λ (nm) | T_out sampled peak λ (nm) |
|---|---:|---:|---:|---:|
| S1 v3 toy, final forward | 0.93461001685 | 0.88977241403 | 1560.9639846 | 1560.9839857 |
| S2 recentered gate point | 0.97181021498 | 0.92574850696 | 1560.8782190 | 1560.8702190 |

The S1 reconstructed T_out objective equals its original JSONL FOM
0.8897724140312584 to floating-point precision. Its original reported
`t_pk=0.9346025447026274` is the sampled modal peak. Thus the discrepancy is
real and explained by different observables, not stale optimizer logging.

Installed lumopt2 `PortResults` takes `abs(T_out)`. The callback reports
`abs(S21)**2`. Ansys defines S using the ratio of modal amplitudes and powers;
reconstructing `abs(a2/a1)**2 * abs(N2/N1)` reproduces the exported S power
to <7e-16 on all three recovered spectra. The two metrics also have different
spectral shapes: the ratio is not a universal scale factor. Do not "correct"
old FOMs by adding 0.045 or dividing by 0.95. Even `T_out/T_in` differs from
modal S power by ~0.0047–0.0049 at these peaks. The precise decomposition of
that smaller difference into finite-grid corrections is not established here.

New callback fields explicitly label/report both definitions, their peaks,
peak wavelengths, the ratio at the modal peak, and source-port T_in.
The objective is retained as T_out for recovery; an optional user preference
question about a future modal objective was sent. Switching the objective
would require a consistent adjoint derivation, including source normalization.

**Confirmed implementation bug fixed:** the wavelength selector adjoints
differentiate T_out, but `_wg_dTp` previously used spectral slopes of modal S
power. It now uses T_out slopes. On the stored S1/S2 spectra the ratio of these
denominators is 0.94623 / 0.95239, i.e. a ~5% scaling mismatch. A non-unit
normalization regression drives the actual callback and fails the old code.
This fixes the mixed normalization. The method is still a local translating
lineshape estimate, not an exact derivative of the modal resonance when
normalization varies with geometry. A future campaign needs a hardware smoke
for this change; the current calibration runners do not exercise the optimizer.

Primary references:
- [Ansys FDTD ports](https://optics.ansys.com/hc/en-us/articles/360034382554-Ports-FDTD-Simulation-Object): S modal normalization and T_in/T_out direction conventions.
- [Ansys sourcepower](https://optics.ansys.com/hc/en-us/articles/360034925313-sourcepower-Script-command): source-power normalization of transmission.
- Installed `lumopt2/fom/simulation_results.py`, `port_fom.py`, `base_fom.py` and `core/project.py`, inspected read-only.

## Surviving evidence and access limits

Old Athena root:
`/home/evyatarrubin/bragg_sim_athena/results/validate_te/results/validate_te_s2/`.

- `lumopt2_te_s2_cport_fd_c_files/fwd_default.fsp` and
  `adj_default_Port_2.fsp` have companion HDF5 fields and successful solver logs.
- The equivalent `_cport_im_c_files/` pair also survives.
- `fwd_default_fd_p0_neg.fsp` is complete: successful solver log, 10156.6 s
  walltime, autoshutoff 9.96e-8, 6.74e-11 s simulated time.
- `p0_pos` is incomplete (last log entry 2.82%, no success marker); `p29_pos`
  was initializing; remaining prepared scenes are unsolved. File existence
  and size alone were insufficient to establish completion.
- The **S1 anchor companion HDF5 is missing**. Its copied FSP cannot provide
  port expansion results locally. The S1 toy companion exists and was used.
- Athena and IGUM SSH were accessible. Local hidden Lumerical CAD was
  accessible. No local FDTD solve or vendor-installation modification occurred.
- Scratch downloads/builds/logs remain under
  `C:/Users/evyat/AppData/Local/Temp/codex_s2_recovery_wynjkxox/` and
  `.../te_recovery_build_caf83s_v/`. The first build probe's generic monitor
  checker expected a width-adjoint twin absent from a port-only spec; the
  corrected port-only build check passed. This was a checker mismatch, not
  evidence of a missing required port-calibration monitor.

## Recovery implementation

`runners/lumopt2_design/recover_te_s2.py` is an isolated, one-leg-per-task
recovery study. It does not alter or resave old solved projects.

| Task | Action |
|---|---|
| 0 | Export stored spectra; completed as 170745_0 |
| 1..12 | ±2 nm FD legs: corr_1, corr_30, avg_1, shift_1, shift_30, cavity width; positive then negative |
| 2 specifically | Reuse the existing successful negative corr_1 solve |
| 13,14 | Reconstruct full-precision Re/Im gradients from existing broadband forward/adjoint fields, with no engine solve; compare with original printed gradients |
| 15 | Fit the original softmax objective after all evidence is present |

Each completed task publishes an atomic JSON checkpoint with specification,
parameter vector and source-code hashes. A restart skips completed matching
tasks. Identity mismatches or malformed checkpoints fail loudly. Each failed
attempt gets its own directory; all original results and new attempts survive.
At worst, preemption loses one solve rather than a whole 14-solve batch.

New FD-only scenes disable `optimization_dft` recording after `set_fwd`.
This monitor is not needed for scalar finite differences. A hidden CAD check
proved exact equality of all three locked mesh arrays before/after disabling
it: 2676 x points, 69 y, 70 z. Source/ports, field_profile and other monitors
remain present; the adjoint twin is absent by design in this port-only spec.
The built ports select fundamental TE and lie outside the grating. The runtime
repeats the mesh equality check, requires a successful solver-log marker and
rejects missing/nonfinite/dead/off-window spectra.

Stored HDF5 allocated dataset sizes: optimization monitor 5.229 GB, all other
monitors together ~0.772 GB. Eleven new FD solves should therefore add roughly
8.5 GB instead of ~66 GB. Initial quota was 252G/300G soft/330G hard; no cleanup
was performed. All runs preserve HDF5 (`KEEP_H5=1`).

`recover_te_s2_peak.py` separately checks the **actual v3 peak3 objective**:
tasks 0/1 reweight the saved broadband fields with its Jacobian; task 2 fits
against the peak3 scores of those same twelve FD legs and also writes the
legacy softmax fit. Vendor source inspection verifies that the port adjoint
source is independent of these objective weights. This avoids additional
FDTD solves. The peak fit requires rank 2, matching signs, <=10% residual
per class and <=10% leave-one-out errors. Coefficients are reported, never
installed automatically. A failed gate leaves the campaign blocked.

The older `run_validate_gradient` remains non-resumable; use these task arrays
for this recovery. They are not a generic repair of every historical caller.
Do not edit the fingerprinted physics/recovery sources during the queue;
new source identities intentionally reject old checkpoints.

## Validation and dispatch

- Existing six offline gates passed after the callback correction.
- Recovery suite: nine tests passed, including interruption/retry, immutable
  checkpoint publication, identity mismatch, malformed/incomplete data,
  no-CAD completed-task resume, rejecting a bad fit, and saved-field peak
  reweighting without any solve/save API available in the test double.
- The recovery gate is now included as the seventh offline gate.
- Hidden local port-only CAD build/mesh equality passed; no local solve.
- Run tests with UTF-8 output on Windows (`python -X utf8 ...`); the existing
  fit helper prints Unicode verdicts. A first negative-control test run hit
  the Windows cp1252 console encoding; rerunning under UTF-8 passed.
- Study-list generation/import and Git whitespace checks passed.
- Athena preflight: empty queue before launch, both license TCP ports open,
  reliable IGUM lmstat reported 50 issued/0 used. One A100 solve budgets 7
  standard seats. Arrays are serial and chained with afterok dependencies.
- The 2h_2g lane rejected the 256G peak-analysis submission with
  `QOSMaxMemoryPerUser`; it was retried on the compatible 12h_4g lane with
  the same requested memory and 2-hour walltime. No job was created by the
  rejected submission.

Current job IDs and the final monitoring snapshot are appended below. The
full calibration is **not complete until the queued numerical results pass**.
Do not translate a successful submission into a scientific PASS.

### Queue snapshot, 2026-10-06 approximately 21:41 Israel time

| Job | Tasks | Resources / dependency | Status at snapshot |
|---|---|---|---|
| 170745 | 0 | 32G, 30 min | completed extraction |
| 170746 | 1,2,13,14, throttle 1 | A100, 256G, 6h/task, 12h_4g | task 1 running; others await array slot |
| 170750 | 3..12, throttle 1 | same; afterok 170746 | dependency pending |
| 170752 | 0,1 of peak study, throttle 1 | A100, 256G, 2h/task, 12h_4g; afterok 170750 | dependency pending |
| 170754 | 2 of peak study | 8G, 15 min, 2h_2g; afterok 170752 | dependency pending |

All dependencies, task throttles, memory and walltimes were checked using
`scontrol`. The deployed engine and both recovery runner hashes matched the
local files. Total planned new FDTD solves: **11**, reusing the twelfth FD leg
and all four original forward/adjoint solves. Expect many hours to roughly a
day or more, with uncertainty from hardware speed, queueing and preemption.
One interrupted solve may need repeating; there is no engine-level checkpoint.

Task 170746_1 runs on n307 (internal Slurm JobId 170747). At ~11 minutes it
was still in frequency-dependent source initialization. A read-only process
check showed fdtd-engine using ~99% of one CPU, ~568 MB RSS; GPU utilization
was 0% with 469 MiB allocated. No solver error was reported. A log comparison
identified **one CPU thread** (`-t 1`) versus **eight** in the original completed
leg (`-t 8`, initialization 302s). Both jobs allocated eight CPUs. This is a
resource-default performance bug, not evidence of a license hang. The wrapper
now explicitly sets/readbacks threads from `SLURM_CPUS_PER_TASK`, after
selecting GPU, without changing MPI processes. The active solve is preserved;
the correction applies to later CAD sessions. The matching IGUM wrapper was
edited locally only. See final notes below for deployment/verification.

Read-only local watcher:
`athena/scripts/watch_te_recovery.py`, launched hidden as Windows PID 42156.
It polls Athena every 15 minutes, fetches only new scalar checkpoints, records
job-state/checkpoint changes and stops on failure/completion/SSH error or at
72 hours. Files go to
`results_from_athena/recover_te_s2_20261006/watch/` (`events.jsonl`, state
snapshots, logs and checkpoint copies). It does not send messages, modify
server state, or automatically repair failed jobs. It needs the laptop awake;
the server job chain does not. Check the watcher's process/logs before assuming
monitoring is still active. Initial collection was verified locally.

Final numerical outputs, once completed:
- `/home/evyatarrubin/bragg_sim_athena/results/recover_te_s2/results/lumopt2_te_s2_cport_recovery_20261006/task_15.json` (softmax fit).
- `/home/evyatarrubin/bragg_sim_athena/results/recover_te_s2_peak/results/lumopt2_te_s2_peak_calibration_20261006/task_02.json` (peak3 fit and held-out verdict).

Use the full-precision JSON coefficients, not the six-decimal printed tuple.
If an upstream task fails, downstream afterok jobs may remain pending with
DependencyNeverSatisfied. Inspect the cause before any resubmission or
dependency change. No cancellation or cleanup was authorized by this workflow.

### Thread fix verification/deployment

Both cluster wrappers passed an offline call-path test for an eight-thread
allocation and for rejecting an incorrect readback, without setting MPI
processes. Hidden local CAD accepted `setresource('FDTD',1,'threads','8')`
and returned eight; its original setting was restored. The Athena wrapper
was atomically published after hash verification, preserving the prior copy
as `/home/evyatarrubin/bragg_sim_athena/scripts/athena_run_one_before_threads_20261006.py`.
Published SHA256: `03a23e518561a5bad45c93dc92c0235d1bd841f0605291c1bb917d9bc218f0fc`.
The current engine command line is already fixed at one thread and was not
interrupted. Check the next task's resource readback and engine log for `-t 8`.
The speedup itself has not yet been measured. IGUM was not updated remotely.
See [Ansys resource configuration](https://optics.ansys.com/hc/en-us/articles/360058790674-Resource-configuration-elements-and-controls)
for the distinction between processes, CPU threads and GPU solver work.

### Readiness follow-up, October 6 at 22:07 Israel time

See `INVERSE_DESIGN_READINESS_2026-10-06.md` for the deeper source/seed audit,
newly reproduced restart hazards and successful native IGUM CAD check.
Fingerprint-bearing recovery/physics source files were not changed.
170746_1 is still running on n307, now past source initialization and into
GPU timestepping: latest 2.9602 ps, autoshutoff0.00177781. The extraction
task_00 is the only scalar checkpoint; the other16 chain tasks are pending
on array throttling or dependencies. No scientific calibration verdict yet.

Local Athena DNS failed once; verified IP132.68.1.206 worked while retaining
HostKeyAlias=athena.technion.ac.il and strict key checking. Watcher was
hardened against missing accounting rows/query failures and tested (4pass).
The local watcher42156 was replaced with **PID42384**, using that address;
old logs remain and new logs are `watch/watcher_v2_{stdout,stderr}.log`.
No Slurm job was cancelled, no physics source uploaded, no files deleted.

### Readiness status check 2026-10-06T22:50:26+03:00

170746_1 and 170746_2 completed successfully (exit0). Both p0 finite-difference legs now have immutable scalar checkpoints: one new solve and one reused original. Task170746_13 is reconstructing the real gradient from saved fields on n306; task14 and the downstream chain are pending (14 tasks pending total). Thus2/12 FD legs are available, with no final calibration fit yet. Checkpoints task_00, task_01 and task_02 were fetched by the read-only watcher. New task13 CAD logs verify the thread correction: FDTD threads=8, GPU selected, build8.35.4572. This verifies resource setup, not yet an eight-thread solver command or measured speedup. The native container wrapper printed fuse-overlayfs shutdown timeout notices after tasks1/2, but Python and Slurm both returned exit0 and the result checkpoints are present.
