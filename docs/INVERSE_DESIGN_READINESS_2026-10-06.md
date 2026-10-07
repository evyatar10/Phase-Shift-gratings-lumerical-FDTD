# Inverse-design code, seeds and server readiness

This is a continuation map, not a claim that every historical study has been
physically validated. Read `AGENTS.md`, then `TE_RECOVERY_2026-10-06.md` for
the authorized, active S2 recovery. The user retains TE, N=98 per side,
60 free periods per side, no posts, and each seed's own approximately 20 µm
width specification. No new optimization campaign was launched in this audit.
The user said “NIGOM”; IGUM is the identifiable second cluster. A question
about whether this means a third server remains unanswered.

## Code map and review coverage

| Layer | Files / entry points | Contract and review scope |
|---|---|---|
| Machine paths | `config.py` | Server wrappers replace BASE_SAVE_DIR, LUMAPI_PATH, NEFF_DATA_PATH; derived layouts/results paths follow RUN_NAME. Reading a derived path creates its directory. |
| Physics configuration | `simulation_config.py` | Dataclasses -> `to_device_kwargs()` -> builder. SI units except explicitly nm fields. Arbitrary Python attributes can silently do nothing. Conversion path reviewed; not every optional device branch tested. |
| Study inputs | `experiment_card.py`, `runners/sweeps/sweep_spec.py` | Cards and zipped/cartesian sweeps. Mapping reviewed; a known cavity-length no-op is reproduced below. |
| Geometry / Lumerical scene | `bragg_device.py` | Region, parity, pitch-aligned mesh, geometry, source/ports, monitors, scan. TE seed and perturbed scenes checked with CAD in the initial audit; exotic paired/metal/asymmetric shapes not all revalidated. |
| Production simulation | `runners/single/run_simulation.py` | Build -> update scan -> apply monitors -> save FSP -> solve -> post-processing -> cleanup. `build_cfg()` carries server-visible overrides; edits only in `__main__` do not. |
| Measurements | `sim_helpers.py`, `post_processing.py`, `analysis.py` | Resonance selection, y-integrated field envelope, spectra, phase corrections and exports. Critical width/spectral paths reviewed. Spatial and spectral FWHM are different observables. |
| Current inverse design | `runners/lumopt2_design/lumopt2_design.py` | CampaignSpec, parameter layout/map, project generation, objective, field/port adjoints, logging, guards, restart, optimizer. Critical TE paths reviewed against installed vendor code and existing gates. Historical penalty and TM branches are retained. |
| Current optimizer math | `runners/lumopt2_design/v3_step.py` | Frequency-grid parabola peak, constrained bounded step, restoration and radius control. Covered by existing offline gates; mathematical tests do not certify physical gradients. |
| TE seeds | `campaign_te_s1.py`, `campaign_te_s2.py`, `runners/sweeps/itai_hh_nt60w20.py` | Source arrays, bounds and anchors traced and checked. See table below. |
| Validation / recovery | `validate_te.py`, `recover_te_s2.py`, `recover_te_s2_peak.py` | Task maps, saved fields, FD legs, calibration and checkpoints reviewed/tested. Recovery currently has Athena-specific OLD paths. |
| Other optimization families | `runners/inverse_design`, `gradient_free_design`, `fd_gradient_design`, `lumerical_native_optimization` | Separate legacy lumopt, PSO, finite-difference and native optimizers. Inventoried; no claim of full behavioral coverage. Do not use their parameter slices for TE lumopt2. |
| Other studies | `runners/tm`, `sweeps`, `scatterers`, `metal_mirror`, `experiment_comparison` | Study definitions and dependencies inventoried; historical measurements require their own numerical provenance. |
| Convergence / predictive tools | `convergence_testing`, `python_tools` | Mesh, box, shutoff, radiation, CMT and Q3dB work. Prediction is distinct from an optimizer's measured width constraint. Not all analyses rerun. |
| Plotting | `matlab_plotting`, `matlab_analysis` | Inventoried, not executed wholesale. MATLAB entry points and legacy scripts are not all runtime-certified. |
| HPC | `athena/`, `igum/` deploy + jobs + scripts | Both launch chains and environment differences reviewed; live checks below. `zeus/` is a legacy PBS path, not a verified third current target. |

`python_tools/audit_readiness.py` provides a repeatable, non-solving inventory:
source hashes, symbols, Python syntax, seed contracts and profile arrays. Its
first run found **380 Python files, 58,160 Python lines; all compiled in memory**.
It also inventoried 23 shell and 128 MATLAB files. The shell line count includes
the 566,609-line bundled `container/AnsysInstaller.sh`; that is vendor installer
payload, not code reviewed line by line. Counts change as files are added.
The manifest is evidence of inventory/syntax, not a statement that all those
files have been semantically reviewed or executed.

`best_designs.py` contains a non-UTF-8 byte in a comment. Python's byte-source
compiler accepts it, and the seed imports work, but strict UTF-8 text tooling
fails. Do not decode the whole mixed-encoding file as cp1252 and corrupt its
existing UTF-8 text. This is a tooling defect, not a failed seed import.

## Uniform, nonuniform and overshoot

S1 and S2 are two starting basins of the **same** TE inverse-design engine.
“Uniform” describes S1's starting profile; its optimized inner teeth become
nonuniform. S2 begins nonuniform and has an intentional corrugation overshoot.
Neither is the old 191-slot TM/comb campaign.

[Seed profiles and stored S1 geometry](readiness_evidence_20261006/te_seed_profiles.png)
show the difference directly. Plot provenance and the source/seed inventory
are in `readiness_evidence_20261006/`; the figure uses existing data only.

| Property | S1 uniform | S2 Itai overshoot |
|---|---:|---:|
| Periods per side / free inner periods | 98 / 60 | 98 / 60 |
| Pitch | 500 nm | 491.06 nm |
| Bulk full corrugation (wide minus narrow) | 250 nm | 494.2 nm |
| Initial inner corrugation range | 250 nm constant | 0–634.3 nm |
| Initial inner mean width | 800 nm constant | 951.4–1030.35 nm |
| Cavity width | 800 nm | 951.4 nm |
| Spatial envelope FWHM anchor | 19.121 µm | 19.636 µm |
| Physical ±2% width band | 18.73858–19.50342 µm | 19.24328–20.02872 µm |
| Single-wavelength twin soft-width anchor | 18.738 µm | 19.558 µm |
| PVA scan center / width / samples | 1560.936 nm / 10 nm / 501 | 1560.464 nm / 2 nm / 501 |
| v3 wavelength trust bound per step | 0.25 nm | 0.05 nm |
| Corrugation physical bounds | 100–450 nm | 0–750 nm |
| Mean-width physical bounds | 775–825 nm | 900–1100 nm |
| Cavity-width physical bounds | 750–1150 nm | 850–1250 nm |
| Port calibration | (0.945335, +0.117012) | **None: recovery active** |
| Field calibration | (0.966720, +0.036560) | (0.983623, −0.088013) |

Both have h=350 nm, constant indices 1.97 / 1.444, no surrounding posts,
PVA optimization numerics, four tiled width-adjoint sources, and width-row
reuse disabled in SPEC_V3. The optimizer adds its configured inner margin
to the physical width band; do not confuse that with changing the user's spec.

The 296-slot vector is `[corr60, avg60, shift60, radius57, x57, distance1,
cavity_width1]`, all in nm. Only 181 slots affect a bare device. Narrow/wide
widths are `avg ∓ corr/2`. The parameter map walks inward from fixed outer
edges; shift shortens each narrow segment and the cavity absorbs
`2*sum(shifts)`. Thus a shift is not an independent translation of one tooth.
The outer 38 periods on each side retain the seed's bulk geometry.

S2's imported table has 61 entries; the 61st is already the uniform bulk
value. Using the first 60 as free parameters and bulk thereafter preserves
the original seed. The source is Itai's reoptimized native-20-µm design,
not his older 15.1-µm design multiplied by 0.562. The corrugation overshoot
peaks at inner period 25; mean width also varies to compensate index.
The zero-corrugation first period is lifted to 5 nm only at the calibration
gate point. Calibration also perturbs shifts/cavity width: it is not an
unmodified seed measurement.

Latest stored S1 row: modal T=0.9527603093, spatial FWHM=19.38500619 µm,
lambda=1561.0439896 nm, objective=0.908596631. Source:
`results_from_athena/campaign_te_s1/lumopt2_te_s1_v3_evals.jsonl`.
This is an intermediate feasible design, not a convergence certificate.
The initial audit records the width-proxy residual and wavelength-step
warning. S2 has no completed inverse-design campaign in this recovery.

## Objective, gradients and monitor commands

The installed API used for source inspection is
`C:/Program Files/Lumerical/v261/api/python/lumopt2/`, build 8.35.4572.
Current online API documentation is useful, but installed-version behavior
and experiments take precedence over assumptions about a newer release.

| API / command family | Project use / invariant |
|---|---|
| `lumapi.FDTD(hide=True)` / `close()` | Licensed headless CAD; opening a session is distinct from an FDTD solve. |
| `addfdtd`, `addrect`, `addpoly`, `addcircle`, `addmesh` | Domain, waveguide/teeth/variants, posts and mesh. SI dimensions in the CAD properties. |
| `addport`, `setnamed('FDTD::ports', ...)` | Port_1 forward source, Port_2 backward receiver, fundamental TE. Ports outside the grating; mode direction controls in/out expansion semantics. |
| `addprofile`, `addfieldregion`, `addimportedsource` | Broadband production profile versus explicit single-wavelength adjoint twin and tiled sources. A DFT monitor is not interchangeable with a FieldRegion source. |
| `set`, `getnamed`, `setnamed`, `getnamednumber` | Select/read/modify objects; assert name mapping and built extents rather than trusting requested config alone. |
| `setglobalsource`, `setglobalmonitor`, `update_scan` | Port grid is uniform in frequency. Twin: source limits off, wavelength center pinned, span zero, one point. Broadband profile is the width authority. |
| `switchtolayout`, `save`, `load` | Layout edits and persisted projects. Switching to layout can discard in-memory results. Saved FSP alone can lack required data: preserve companion HDF5. |
| `getresult(..., 'expansion for port monitor')` | Read S, T_out, T_in and mode normalization. Objective is abs(T_out); logged modal transmission is abs(S21)^2. |
| `getresult('field_profile', 'E')` | Integrate all electric-field components over y, crop to grating, then envelope/FWHM. Raw on-axis width is not this observable. |
| `setresource/getresource` | Select/read back GPU and allocated CPU threads. Do not conflate threads with MPI processes. |
| `project.generate`, `set_fwd`, `run_forward`, gradient field assembly | Generate checks scene/plumbing; a successful generate is not a solve or gradient calibration. |

See [Ansys getresult](https://optics.ansys.com/hc/en-us/articles/360034409854-getresult-Script-command)
and [PortResults API](https://lumerical.docs.pyansys.com/version/stable/api/lumopt2/_autosummary/lumopt2.fom.simulation_results.PortResults.html).

The v3 objective interpolates the resonance peak using three frequency-grid
samples; the old softmax objective has a different Jacobian. Port and width
gradients are separately calibrated. Width's moving-resonance derivative
uses the fixed-wavelength row plus measured `c_W*gLambda`, subject to validity
checks. The bounded QP enforces parameter bounds, width and wavelength bands
with a single step radius; measured trial results still decide acceptance.
The metric is squared bound half-range, so it strongly weights wider blocks.
Changing that metric is a design choice requiring comparison, not a silent fix.

The FOM/modal-transmission discrepancy was reproduced exactly from stored
spectra; it is not a missing scalar conversion. The October 6 callback fix
uses T_out slopes for the wavelength derivative denominator to match its
adjoint numerator. This remains a translating-lineshape approximation;
the corrected optimizer path still needs a hardware smoke before a campaign.

## Both server paths and current checks

| Item | Athena | IGUM |
|---|---|---|
| Login | evyatarrubin@athena.technion.ac.il | evyatarrubin@132.68.58.101 |
| Base | /home/evyatarrubin/bragg_sim_athena | /home/evyatarrubin/research/bragg_sim_igum |
| Runtime | Apptainer + NVIDIA, fixed lumerical-2026R1.sif | Native RPM-extracted R1.3 |
| Lumerical inside runtime | /opt/lumerical/v261 | /home/evyatarrubin/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261 |
| Display/libraries | Xvfb, container binds, TBB preload, CUDA compatibility handling | Qt offscreen, LUM_HOME/lib plus WORK_DIR/scilibs, TBB preload, fabric guards |
| Scheduler | SLURM GPU/QOS lanes | SLURM, required acct-lumerical, paired partition/QOS |
| Current placement | Recovery on A100, 256G, serial dependency chain | No active user jobs in this check; two test-only requests accepted |
| Storage snapshot | 252G used / 300G soft / 330G hard home quota | research 124T shared / 24T available; not a user quota |

IGUM `sacctmgr` still fails to reach slurmdbd on localhost. This does not
mean submission is blocked: `sshare -U`, `scontrol show assoc_mgr flags=qos`
and `sbatch --test-only` work. Both qos-preempt and qos-lumerical report
REQUEUE and 10080-minute MaxWallPJ. The cluster uses preempt/qos;
partition-level OFF alone is misleading. See
[Slurm preemption](https://slurm.schedmd.com/preempt.html).

IGUM engine `-v` failed without its library environment (libglut.so.3), then
passed with the actual job environment: build 8.35.4572, Python 3.13.1,
NumPy 2.2.2, SciPy 1.14.1, lumapi and lumopt2 imports. A first CAD probe timed
out at 55 seconds; its orphan CAD exited before a cleanup signal was needed.
A bounded retry with the complete job environment opened hidden CAD in six
seconds, built an unsaved dummy region/rectangle, read back GPU / 8 threads,
restored the prior resource settings and closed normally (exit 0). No solve
or project save. The exact cause of the first timeout is not established;
do not claim a measured causal fix from the retry alone. The license helper
was still briefly visible after CAD closed; no broad process cleanup was used.
No fresh IGUM compute-node TE solve was run. Its remote wrapper has not yet
received the local CPU-thread correction.

Athena hostname DNS failed locally once. IGUM resolved it to 132.68.1.206;
SSH to that address succeeded using `HostKeyAlias=athena.technion.ac.il`
and `StrictHostKeyChecking=yes`. No SSH configuration or host keys changed.
At 21:53 Israel time job 170746_1 was RUNNING on n307; the rest of the chain
was pending correctly. Only extraction task_00 existed then. The active
one-thread solve was preserved. The published wrapper fixes later sessions.

## Launch and validation procedure for the next session

1. Read the latest state and verify no requested work is already running.
   Do not resend this recovery array. Read source task IDs, dependencies,
   immutable checkpoints and successful solver logs before recovery actions.
2. Use `C:/Program Files/Git/bin/bash.exe` on Windows. Python3.14 is local;
   servers use bundled3.13. Run `python -X utf8 python_tools/audit_readiness.py`
   for inventory and `python -X utf8 runners/lumopt2_design/gates/run_all_gates.py`
   for the seven offline gates. Gates passed in the preceding recovery work.
3. Build/monitor changes need hidden CAD `gate_te_local.py --generate`;
   those TE seed/perturbed equivalence and monitor checks already passed in
   the initial audit. Do not repeat expensive verification without a reason.
4. Before dispatch, check shared licenses, quota, scheduler, versions/hashes,
   runtime, required memory and restart persistence. IGUM test-only accepted
   `--account=acct-lumerical --partition=part-preempt --qos=qos-preempt
   --gres=gpu:1 --cpus-per-task=8 --mem=256G --time=00:05:00 --wrap=true`.
   This validates the request only, not the software on an allocated node.
5. Deploy contract: `--lumopt2-design=runners.lumopt2_design.<module>`,
   `--array-tasks=<ids>`, `--max-concurrent=1`, `--keep-h5`; Athena supports
   `--gpu=a100`. Resource overrides are LUMOPT2_QOS, LUMOPT2_TIME, SBATCH_MEM.
   IGUM does not accept Athena's GPU-selector flag. Verify final partition
   matches QOS; changing QOS alone does not move partitions.
6. `--upload-only` is a real remote source mutation. Both deployers preserve
   remote-only files but replace matching ones. Freeze fingerprinted recovery
   sources until completion; review source drift before any upload.
7. Server path: build_sweep_list.py -> numbered task manifest -> sbatch
   run_python_array.sh -> athena_run_one.py -> study.main(task_idx).
   `validate_te.N_TASKS=40` includes unused indices: read actual dispatch
   branches. Offline list generation does not prove all advertised IDs work.
8. S2 must pass legacy reconstruction and peak3 per-class/held-out gates,
   then its corrected hardware smoke/toy before a long campaign. Never paste
   S1 coefficients into S2 or treat a generated project as calibrated.

## Confirmed open bugs / limitations

| Finding | Evidence / effect | Next action |
|---|---|---|
| Campaign profile filenames restart at ev0000 | Callback writes `<label>_evNNNN.npz`; evaluation IDs restart after recenter/requeue. Later profiles can overwrite earlier ones. Gradient filenames already include timestamps. | Add immutable evaluation identity and persist profile path in each row; update MAC lookup together, before next campaign. Active recovery does not use this callback. |
| Truncated JSONL prevents campaign restart | `_best_from_log` and `_row_of_params` parse every line; valid row + incomplete last row reproduces JSONDecodeError. | Preserve original; support a recoverable final partial record, reject corruption elsewhere. Do not truncate evidence in place. |
| Corrupt optstate silently resets state | `_load_opt_state` returns {} for OSError/ValueError. Losing ineligible-trial history may change restart eligibility. | Distinguish missing state from corrupt state and fail clearly / recover a verified backup. |
| Absolute cavity-length card silently ignored | ExperimentCard(cavity_length_nm=300).to_config().to_device_kwargs() returns override_cavity_length_nm=False. | Until corrected consistently for cards and sweeps, use cavity_neg_detuning_nm = pitch_nm/2 - desired_length_nm. This is not used by TE's parameter map. |
| Envelope with 2–3 peaks fails cubic interpolation | Three-peak synthetic input reproduces scipy ValueError in extract_envelope_peaks. | Add an explicit sparse-profile policy; do not silently fabricate a physical width. Current long TE gratings have many fringes. |
| Mixed encoding in best_designs.py | Byte compilation succeeds; strict UTF-8 read fails. | Repair only offending comment bytes, preserving numerical arrays and existing UTF-8. |
| S2 calibration / corrected optimizer validation | Jobs still running; C_port=None; no corrected optimizer hardware smoke yet. | Continue authorized chain, review measured errors, then smoke/toy. |

These source changes are deliberately not mixed into fingerprinted active
physics. No files/results were deleted and no Slurm job was cancelled.
The read-only watcher was hardened: scheduler command errors cannot masquerade
as completion, and all 17 expected tasks must be completed. Four new offline
watcher tests passed. It now accepts an optional verified SSH address while
retaining hostname host-key checks. It is not a scientific fit validator.

Raw audit evidence: `C:/Users/evyat/AppData/Local/Temp/codex_readiness_rxu5yhn0/`.
Small bug reproductions: `C:/Users/evyat/AppData/Local/Temp/codex_readiness_repros_bf4m2sox/`.
Initial monitor/build evidence: `C:/Users/evyat/AppData/Local/Temp/codex_te_audit_ojib8204/`.
The repository and two Codex skills preserve the operational knowledge; raw
local temp evidence is supplementary, not the only handoff.

### Final check, October 6 at 22:07 Israel time

Athena 170746_1 advanced past initialization into GPU timestepping. Its last
log showed 2.9602 ps simulated and autoshutoff 0.00177781 (not yet the 1e-7
termination threshold). Only task_00 was checkpointed; all other tasks were
pending behind the intended array/dependency limits. Engine maximum-time
remaining is an estimate to the full 2000 ps cap, not a completion forecast
for an autoshutoff-terminated solve.

The updated read-only watcher is running as local PID **42384** (started
22:06:59); previous watcher42156 was explicitly replaced after checking its
command line. Original logs were preserved; new logs are watcher_v2_stdout.log
and watcher_v2_stderr.log in the same watch folder. It uses the verified
Athena address with hostname key verification. No server job was interrupted.
Both Codex skills validated. All15 Athena/IGUM shell scripts passed Bash
syntax checks; `git diff --check` passed (line-ending notices only).
