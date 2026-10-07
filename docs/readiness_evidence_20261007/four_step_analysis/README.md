Current correction: use `existing_evidence_pins_role_bound.json` with the
current helper. The original manifest is retained unchanged as evidence of
the earlier failed comparison. Read HISTORICAL_IDENTITY_DIFF.md for the
exporter-versus-forward-source correction and the measured four-step result.
Current actual report: ../s2_corr30_four_steps_measured.json. Both original
and new source roles remain qualified; no calibration pass is claimed.

# Four-step comparison helper — offline preparation

No repository, server, CAD, solver or calibration changes. Six focused synthetic
tests pass, exit0; exact commands/hashes are in `focused_check_results.json`.
`existing_input_readiness.json` records that the real existing center/half-step
inputs and review marker were accepted, then the first missing convergence
checkpoint stopped analysis. No real four-step report has been produced.

## Input filenames

The evidence directory must contain these **raw original checkpoints** and
matching unmodified ports files:

```text
s2_fdstep_task00_original.json       s2_fdstep_task00_ports.npz
s2_fdstep_task01_original.json       s2_fdstep_task01_ports.npz
s2_fdstep_task02_original.json       s2_fdstep_task02_ports.npz
s2_corr30_convergence_task00_original.json  s2_corr30_convergence_task00_ports.npz
s2_corr30_convergence_task01_original.json  s2_corr30_convergence_task01_ports.npz
s2_corr30_convergence_task02_original.json  s2_corr30_convergence_task02_ports.npz
s2_corr30_convergence_task03_original.json  s2_corr30_convergence_task03_ports.npz
corr30_convergence_parent_review.json
```

Convergence task0/1 are +1/−1 nm; task2/3 are +0.25/−0.25 nm. The existing
task01/02 pair is ±0.5 nm. The original archive's `base_ports.npz` and
`task_03_ports.npz`/`task_04_ports.npz` provide the nominal and ±2 nm pair.

Raw checkpoint/marker bytes matter. Do not pretty-reserialize them before
fetching. NPZ bytes must match the exact result's `ports_sha256`; names alone
are insufficient. Keep FSP/HDF5 provenance inside checkpoints. This helper
does not read remote artifacts or replace the runner's artifact verification.

`existing_evidence_pins.json` binds the already preserved center/half-step
JSON/NPZ, old archive/members, center source map, old checkpoint metadata and
parent marker. SHA256:
`c37f9c7231b25547de9f4645004a5221ab91e6081a578f175848d7745d1bc6bc`.
Existing half-step checkpoints do not contain ports hashes; the manifest
provides their independently preserved byte pins. New convergence endpoints
must pass their saved NPZ hashes and full metadata/score checks. Their raw
checkpoint hashes are recorded in the result, not claimed externally approved.

## Command when every endpoint exists

```powershell
python -X utf8 C:/Users/evyat/AppData/Local/Temp/s2_four_step_analysis_20261007_c317aab2/compare_four_steps.py `
  --evidence-dir C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/docs/readiness_evidence_20261007 `
  --old-spectra C:/Users/evyat/AppData/Local/Temp/s2_peak_failure_spectra_ppyzxc75/stored_port_spectra.zip `
  --pins C:/Users/evyat/AppData/Local/Temp/s2_four_step_analysis_20261007_c317aab2/existing_evidence_pins.json `
  --output C:/Users/evyat/AppData/Local/Temp/s2_four_step_analysis_20261007_c317aab2/four_step_report_NEW.json
```

The output must not exist. Missing inputs fail before publication. Parent owns
future repository integration and any actual evidence report destination.

## Checks and output

Reject wrong task/sign/step/task_map, changed base/endpoint parameters,
different complete specs apart from the intentional label, different source
maps, wrong center or parent-marker hash, different remote provenance paths,
NPZ corruption, and mismatched complete p1/p2 wavelength/frequency grids.
Both saved T_out and modal peak3 values must reproduce within 1e−12, solely
an arithmetic tolerance. Old ±2 metadata is verified against its original
center/source family; its old engine map legitimately differs from the new
center and is not silently conflated with it.

For h=2,1,0.5,0.25 report central/odd/even/second differences with **both**
nominal center values; center peak3 Jacobians applied to spectral FD versus
moving-peak FD; full spectral odd/even/derivative vectors and adjacent halving
differences; observed scalar difference ratios and conditional Richardson
values. Old/new nominal differences are retained. No ratio is declared a
convergence order and no noise bound, global fit, coefficient, spatial-width
qualification or automatic calibration pass is inferred. The verdict remains
`DIAGNOSTIC_ONLY_OVERALL_CALIBRATION_REMAINS_FAILED`.

Synthetic cubic/quartic amplitude plus nonlinear peak-motion tests verify
known derivative/curvature formulas, expected halving ratios, conditional
Richardson and fixed-center versus moving-peak differences. Negative controls
cover swapped checkpoints, wrong code/point/center/marker/score, altered NPZ
away from the peak, modified archives, changed grids even after a hash update,
missing data and exclusive output preservation.

Helper SHA256: `592f8c371f4934af1b5b281b6d828d9a301ac42d21458eccc2073c34ac75d52c`.
Tests SHA256: `6fd15f2637326db11c5087cda5b44c7069ad9056c67d6355e88b1a30bc053446`.
