# Codex audit and continuation record — 2026-10-06

## Scope and authorization

The user requested project onboarding, a critical review, validation, safe
bug fixes, and reusable Codex skills ahead of further inverse design.
They explicitly retained TE, N=98 periods per side, 60 free periods per side,
no surrounding posts, and spatial width near 20 µm. Existing per-seed bands
remain unchanged. This session did not restart any simulation.

The supplied memory path was missing a separator. The actual file read is
`C:/Users/evyat/.claude/projects/c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes/memory/project_te_inverse_design_lane.md`.
Its latest stop instruction agrees with handoff §6.2.7 and live accounting.
Claude's local memory, skills and project files are accessible; that does not
provide access to Claude's private reasoning, or automatically port its hooks.

Review concentrated on handoff orientation, recent TE/v3 history, resonance,
cluster operations, live project rules, inverse-design theory and current
runner/engine/gate/monitor code. Historical studies and the large memory dump
are indexed references, not all independently revalidated physics. This is
not a claim to have certified every archived runner or every Lumerical API.

## Verified state

Read directly from `results_from_athena/campaign_te_s1/`:

| Quantity | Last saved S1 campaign row |
|---|---:|
| Row identity | last of 8 JSONL rows; campaign eval 3 (eval numbers restart) |
| Logged modal peak transmission | 0.9527603093 |
| Spatial envelope FWHM | 19.38500619 µm |
| Resonance wavelength | 1561.04398960 nm |
| Spectral FWHM | 1.00723576 nm |
| Loaded Q, derived from the spectrum metrics | 1549.829797 |
| Model-derived intrinsic Q estimate | 64831.2883 |
| Logged optimizer FOM | 0.9085966310 |
| Logged reflection at peak | 0.0024382554 |
| Residual `1-T-R` | 0.0448014353 |
| Saved trust radius | 30 nm |
| Saved counters | n_acc=7, n_rej=1 |

Sources: `lumopt2_te_s1_v3_evals.jsonl`, `*_optstate.json`, `*_proj.jsonl`,
and `*_grads_it003.npz`. These are stored measurements, not new simulations.
The directory contains no raw spectra/field profiles with which to independently
re-extract the reported resonance/FWHM. Scalar Q/loss identities, finite
gradient arrays, seed bounds, and `_best_from_log` selection were checked.
The selected resume vector exactly matches the last saved row (296 entries).
This is preliminary, unconverged PVA evidence; no conformal confirmation of
this S1 candidate was run here.

Read-only cluster snapshot around 20:52 Asia/Jerusalem:

- Athena and IGUM user queues empty.
- Athena `sacct`: 170253_0 cancelled after 09:41:48; 170505_14 cancelled after
  06:14:51; 170505_15 completed after 02:46:24.
- Athena home: 252G used, 300G quota, 330G hard limit.
- IGUM `lmstat`: 50 `lum_fdtd_solve` licenses issued, 0 in use. A snapshot
  is not a reservation; recheck before a future authorized launch.

One initial PowerShell SSH command lost quoting around an squeue format
string; the simpler `squeue -u evyatarrubin --noheader` succeeded. No upload,
submission, cancellation, remote file edit, or cleanup was performed.

## Findings and changes

### Fixed locally

1. **Wrong anchor in copy/paste output.** `validate_te` task 1/11 printed
   `softw_um` as `SOFTW0_UM`, although the width adjoint needs its twin's
   `softw_adj_um`. It now prints the twin and labels the raw value separately.
   Existing S1/S2 constants already use the correct twin values; no campaign
   constants or objectives were changed.
2. **TE bounds missing from the combined offline gate.** The six-gate runner
   did not include the licensed TE gate despite the handoff saying it did.
   `predispatch_check.py` now also checks both TE seeds and FD legs offline.
   The runner clearly distinguishes an offline pass from dispatch readiness.
3. **Missing automated monitor assertions.** `gate_te_local.py` now checks
   built port ordering, TE symmetry, mode selection, port sample count,
   profile coverage including maximum tooth/cavity widths, and explicit
   single-wavelength twin/tile settings. Its default build directory is unique
   and preserved, instead of reusing a fixed temporary path.
4. **Upload could delete remote-only code.** Removed `rsync --delete` from
   the runners/convergence uploads in BOTH cluster scripts. Uploads still
   replace matching files, so running jobs must be considered before upload.
   Retired remote modules now remain until a separately authorized cleanup.
   No deployment of these changes was performed.
5. **Stale guidance.** Added current TE/v3 method to `THEORY.md` §13; corrected
   the `1-T` loss statement and qualified overly broad claims that scalar
   constrained optimization cannot work. Updated current-state pointers,
   license budgeting, and the mistaken GPU-memory/concurrent-FD notes.
   Historical handoff snapshots and measurement files were preserved.
6. **Codex continuity.** Added root `AGENTS.md` plus validated personal skills
   `pi-shift-inverse-design` and `pi-shift-cluster-ops`. These route to live
   project evidence rather than duplicating the entire Claude archive.

### Open before further optimization

1. **S2 cannot start a calibrated campaign yet.** `campaign_te_s2.ADJ_FIX_PORT`
   is still `None`. Its completed Im half must be paired with FD+Re at exactly
   the same operating point/window/numerics. Do not substitute S1's factor.
2. **Long FD validation has no checkpoint/resume.** Current
   `run_validate_gradient` delegates to vendor `validate_gradient`; the vendor
   queues all perturbations and extracts their FOMs after the queue finishes.
   It also strips solved fields from completed FSPs. No per-leg numeric
   checkpoint protects a multi-hour cancellation/requeue. Repair persistence
   (including an identity check and restart tests) or split the work before
   rerunning task 14. No checkpoint repair or hardware test was attempted here.
3. **FOM and reported transmission differ.** The installed vendor source
   `lumopt2/fom/simulation_results.py::PortResults.get_transmission_results`
   uses `abs(T_out)`; our callback uses `abs(S)**2`. This establishes the two
   different inputs, not the physical cause of their roughly 0.045 offset.
   Read both arrays and normalization data from the SAME saved solved project
   before changing the objective/gradient. Peak selection and gLambda also
   need a consistent resonance definition (known review item G7).
4. **Width surrogate warning is real.** Last row `wg_resid_um=0.052445` exceeds
   `WG_RESID_WARN=0.05`. Measured FWHM still satisfies its band. This is a
   diagnostic to investigate, not evidence that the final width is invalid.
   The last twin wavelength lags the recorded peak by about 0.0400 nm.
5. **Step verification is not an unconditional pass.** Derived from the four
   campaign rows and corresponding projection rows:

   | Step | measured/predicted FOM gain | moving-width prediction error (µm) | wavelength prediction error (nm) |
   |---|---:|---:|---:|
   | 1 | 0.98293 | -0.004479 | +0.020610 |
   | 2 | 0.95123 | -0.011703 | -0.015884 |
   | 3 | 0.84251 | -0.018414 | +0.006810 |

   Width errors use `v3_pred_rows[0]`, not fixed-wavelength `dw_pred`.
   Step 1 narrowly exceeds the stated 0.020 nm wavelength-error allowance;
   the measured peak is quantized on a roughly 0.020 nm grid. Do not silently
   round this to PASS or declare a physical failure without resolving sampling.
6. **Shift weighting remains a user decision.** Bound-range-squared metric
   ratio is `(350/15)^2 = 544.444` for corrugation versus shifts. Saved iterate-3
   data have 16 shifts at zero within 1e-8 nm and 57 below 0.01 nm; the handoff's
   "57 at the zero bound" is an approximation, not an exact active-set count.
   Maximum shift is 0.263014 nm. A predicted benefit from reweighting has not
   been measured; no reweighting or negative shift bounds were introduced.
7. **S1 and S2 optimize different width bands.** S1: 18.73858–19.50342 µm;
   S2: 19.24328–20.02872 µm. Keep this distinction in comparisons. For a final
   equal-width comparison, settle a common target or re-trim after optimization.
8. **Physical confirmation remains.** Build checks do not establish domain/PML
   convergence, a correct new gradient calibration, or fabrication performance.
   Retain the PVA/conformal distinction and the open far-field end-loss question.

## Validation performed

- Six offline gates passed before changes and again after the gate changes;
  process exit code 0. V3 comparison covered 216 reference-QP instances
  (worst reported relative objective gap 5.4e-11) and driver failure cases.
- Hidden local CAD checks passed for BOTH TE seeds: seed/perturbed geometry
  vs production builder worst difference 0.0000 nm at printed precision;
  shift gap/edge errors below 3.4e-11 nm. Bounds and source tile sizing passed.
- `project.generate()` passed for BOTH TE v3 variants, using provisional
  build-only anchors. This did not bypass S2's campaign calibration guard.
  Local API reported FDTD 8.35.4572.
- New monitor checks passed against saved, actually built S1 and S2 FSPs.
  Eight negative controls rejected wrong symmetry, port inside the grating,
  too-narrow profile span, and twin source-limits sampling. No FDTD solve.
- Anchor-output regression used a stubbed canary with deliberately unequal
  raw/twin widths; it selected the twin and rejected a missing twin field.
- Both modified deploy scripts passed Git Bash `bash -n`; no live deploy test.
- Both new skills passed `skill-creator/scripts/quick_validate.py`.
- `git diff --check` passed. No commits, staging, branch changes, deletion
  of existing files, or result edits were made.

During the final review, additional untracked `.agents/skills/` copies and
`.codex/hooks.json`/`.codex/hooks/check_py_syntax.py` appeared (creation time
21:04:47). They were not created by this audit's tool calls and were left
untouched. The copied lumopt2 skill includes the current TE correction.
The hook's registration/execution in Codex was not validated; do not infer
remote safety enforcement from its presence. Review these separately before
staging the working tree.

Build artifacts and offline gate log are preserved at:
`C:/Users/evyat/AppData/Local/Temp/codex_te_audit_ojib8204/`.
The build run took several minutes through multiple hidden CAD sessions;
no local FDTD engine solve was launched. No MATLAB figures were changed.

## Monitor and command map for the next session

| Component | Location / relevant API |
|---|---|
| Configuration | `simulation_config.py`; direct unknown attributes can be silent no-ops; verify expanded/built geometry |
| Production builder | `bragg_device.py::build`; `addfdtd`, `addrect`, `addmesh`, `addport`, `addprofile`, `setnamed` |
| Sweep pipeline | `runners/sweeps/sweep_spec.py`, `runners/single/run_simulation.py`, `post_processing.py` |
| Width authority | `sim_helpers.extract_and_process_field_profile`; engine `_line_from_res` and `fwhm_env_of_line` |
| TE parameterized scene | `make_func`, `build_base_fsp`, `make_project`; `addfieldregion`, `getnamed`, `save`, `project.generate()` |
| Results | `getresult('FDTD::ports::Port_2', 'expansion for port monitor')`; `getresult('field_profile','E')` |
| Execution | `run_campaign`/`run_projected`; LocalRunner; `fdtd.runjobs` on allocated compute nodes |
| Far field | `sim_helpers.extract_farfield`, `post_processing.extract_farfield_data`, `python_tools/farfield_surface.py` |

Built S1 grating ends are ±49.125 µm; ports ±59.125 µm. S2 ends are
±48.2466425 µm; ports ±58.0678425 µm. Profiles cover the grating, centered
at y=z=0. TE y-min is Anti-Symmetric and z-min Symmetric. The twin is
single-frequency with source limits disabled. Ports span transversely beyond
the nominal box intentionally; that alone is not evidence of bad placement.
Far-field surfaces are not enabled in these campaign builds, so this check
does not validate their aperture or radiation completeness.

Read the actual parser before launch. Git Bash here resolves Python 3.14,
rsync in `/c/Users/evyat/bin/`, and its own SSH; PowerShell's default bash
points elsewhere. Long TE campaigns use the documented 256G/Athena/QOS
recipe only after authorization and renewed preflight. Use
`--lumopt2-design=<module>` and an explicit task subset where needed. The
runner's `N_TASKS=40` includes reserved/unimplemented indices.

No remaining question blocks reading and local development. Before a future
launch: address S2 persistence/calibration if pursuing S2, decide whether S1
metric reweighting is wanted, investigate the numerical diagnostics, and
obtain the user's requested run/cluster direction. Preserve the existing
S1 state; do not repeat its completed optimization steps from the uniform seed.

## Primary references checked

- [Ansys license consumption](https://optics.ansys.com/hc/en-us/articles/360058577794-Ansys-optics-solve-accelerator-and-Ansys-HPC-license-consumption): standard GPU solve licensing counts SMs in groups of 16; CPU requirements can also matter.
- [Ansys frequency-domain monitors](https://optics.ansys.com/hc/en-us/articles/360034902393-Frequency-domain-monitor-Simulation-object): frequency sampling, source limits, spatial interpolation and PML recording are separate settings.
- [Ansys FDTD ports](https://optics.ansys.com/hc/en-us/articles/360034382554-Ports-FDTD-Simulation-Object): port direction, modal expansion, and monitor sampling settings.
- Installed vendor source was inspected read-only at `C:/Program Files/Lumerical/v261/api/python/lumopt2/`, especially `fom/simulation_results.py` and `utils/fd_grad.py`; it was not modified.
