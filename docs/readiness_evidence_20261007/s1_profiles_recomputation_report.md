# Recovered S1 spatial profiles independently recomputed

All16 recovered original profiles reproduce both logged production envelope
FWHM and second-moment sigma **exactly**, maximum absolute difference0um.
No new FDTD solve was used. Four focused offline test groups pass, including
analytic envelope/MAC cases and invalid-profile/ambiguous-row/pin guards.
The parent independently reran the actual/synthetic script and reports identical
numerical output.

| Saved profile | Associated JSONL row | Recomputed spatial FWHM(um) |
|---|---:|---:|
| S1 anchor | line1 |19.120888652047647|
| S1 v3 toy eval0 | line1 |19.119801343547707|
| S1 v3 toy eval3 | line4 |19.410602967117633|
| S1 campaign eval0 | line5 |19.410526608744398|
| S1 campaign eval1 | line6 |19.398864809463042|
| S1 campaign eval2 | line7 |19.391716718423915|
| S1 campaign eval3 | line8 |**19.38500618568187**|

The campaign profiles share an exact1960-point x grid, -48.975..48.975um
(approximately50nm spacing). Smoke profiles have the separately recorded
N70 cropped range -34.975..34.975um and are not finalN98 confirmation.
All profiles have finite, nonnegative intensity and increasing coordinates.
Logged sigma values reproduce exactly as well.

| Campaign transition | Recomputed intensity-profile MAC | Difference from log |
|---|---:|---:|
|0→1|0.9999946088339488|0|
|1→2|0.9999841830660399|0|
|2→3|0.9999902837222255|0|

MAC uses previous accepted intensity as reference, current intensity interpolated
onto its grid, and the production squared-integral formula. The grids happen
to be equal here. This is intensity-profile MAC, not a reconstructed complex-E
overlap or independent proof that no physical mode change occurred.

Final profile SHA256:
c692c1f624ac69c479815ca1898f503744d884e15490c0f8fdf7d96b3db3f02d.
It uniquely matches label lumopt2_te_s1_v3/eval3/lambda1561.0439895954794nm
to JSONL line8. Remote mtime is8.7221274376s after that row's timestamp.
The associated row has296 parameters with LE-f64 SHA
66e7c00468886c448467ee8432d878eb36ac2451bf2d508608b1dea707be1dd8,
also matching the final gradient archive. NPZ contains only x_um,I,lam_pk_nm;
it has no vector or immutable evaluation ID. Thus this is a strong indirect
row association supported by unique wavelength/time, not direct geometry proof.
Repeated eval numbers were handled: campaign eval0 matches line5, not toy line1;
campaign eval3 matches line8, not copied toy line4.

Production functions used: fwhm_env_of_line and sigma_of_line on saved1D I,
profile_mac on saved pairs. Engine SHA53108bbd368dd699179ab65029628aa4c4b2d8ad687777f88b82f404a0952569;
sim_helpers remains byte-pinned in the numerical report. No sources were edited.
Synthetic20um Gaussian envelope with standing-wave fringes yields
19.999999568274404um (error-4.3173e-7um). MAC is invariant to3x intensity
(0.9999999999999999) and matches a3um shifted sigma4um Gaussian's analytic
exp(-9/32)=0.7548396019890073 exactly. These are arithmetic checks, not physics gates.

Inputs are original-byte pinned by profiles_recovery/pins.json SHA
09998f826d923fd58d923b3e5e236211bd3da4353d63d7e995cd4e7ed40063f0,
under TEMP/te_original_remote_inventory_q8_9_zk0. The full checked report
records every profile's SHA, row/parameter association, remote mtime, grid,
intensity sanity, FWHM/sigma residual and MAC/projection-row correspondence.

Current meaning: this corroborates Claude's saved spatial-width and MAC
numbers from the original processed1D profiles. It does not independently
re-run original full-E y integration, source normalization, geometry, forward
transmission, temporal/mesh convergence, gradient calibration or optimizer.
The final width stays inside retained18.73858..19.50342um band, but remains
above the19.121um anchor (+1.3807%) and does not establish strict no increase.

Parent14:24 raw check found campaign fwd_default_output.h5 absent, its companion
directory empty, and fwd_default_p0.log ending in source initialization without
success. Latest FSPmtime20:25:29 follows accepted row18:50:56 and job170253
cancellation20:32:28. Association with accepted66e7 is unproven; do not extract
that scene as the final spectrum. Evidence is in TEMP/te_s1_final_raw_status_jtdfaudi.
A new candidate transmission remeasurement may become useful, but requires an
explicit current-numerics/timing purpose, preserved vector/spec/readbacks and
runtime/resource review. S2 temporal/gradient remains the critical path.

Reproduce locally (new exclusive output/check directory):

```powershell
python -B recompute_profiles.py `
  --root C:/Users/evyat/AppData/Local/Temp/te_original_remote_inventory_q8_9_zk0/profiles_recovery `
  --pins C:/Users/evyat/AppData/Local/Temp/te_original_remote_inventory_q8_9_zk0/profiles_recovery/pins.json `
  --pins-sha256 09998f826d923fd58d923b3e5e236211bd3da4353d63d7e995cd4e7ed40063f0 `
  --output NEW_profile_report.json --synthetic-dir NEW_synthetic_checks
```

Actual checked output: actual_profile_report_checked.json. The initial and
guarded actual reports are identical. Tests: python -B test_profiles.py,4groups
exit0. No server calls, CAD/solver invocation, repository change or new runner.
