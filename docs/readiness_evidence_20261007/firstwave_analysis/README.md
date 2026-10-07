# First-wave half-class review analyzer

Preparation only. No server access, FDTD/CAD call, coefficient fit, automatic
scientific pass or approval marker is implemented. Last supplied170962 state
was RUNNING08:57; absent local checkpoint files are not evidence it stopped.
The current VPN failure prevents fetching fresh results; this helper does not
retry network access or infer job completion.

Files in this temp package: analyze_firstwave.py, numerical_helpers.py
(copied previously tested arithmetic helpers), test_firstwave.py. Imports the
frozen repository recovery.score solely for offline arithmetic reproduction;
its source hash must match the exact first-wave marker. Python bytecode writes
are disabled. Run from the repository root or supply its absolute --repo path.

Exact new fetch names, kept as original bytes in the evidence directory:

- s2_half_classes_task00_original.json .. task03_original.json
- s2_half_classes_task00_ports.npz .. task03_ports.npz

Existing inputs: s2_fdstep_task00/01/02_original.json and corresponding
*_ports.npz; s2_half_classes_first_review_original.json. Their known immutable
SHA256s are enforced. Center SHA is cd2dd4...e3be4a06, marker6722e188...8d02,
runner4f5f9d21...bba1. The task map is corr30 repeat+/-0.5nm followed by
shift30+/-0.5nm. Every result must match exact full parameter point, full spec,
class/task/sign/step, frozen source code, center and parent-marker identity,
declared artifact family/attempt and FSP/H5 relationship, plus transferred
NPZ hash. All four frequency/wavelength grids must match the center exactly.
All saved recovery.score scalars must reproduce to1e-12. This is an arithmetic
tolerance, not a scientific gate. Remote FSP/H5 contents and solver logs are
not re-opened by this offline analyzer; the runner's checkpoint chain and
parent's preserved original-byte retrieval remain part of that provenance.

Prepare pins NOW from the existing local artifacts, with a NEW output file:

```powershell
python -B analyze_firstwave.py --prepare-pins `
  --repo C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes `
  --evidence-dir C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/docs/readiness_evidence_20261007 `
  --old-spectra C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/docs/readiness_evidence_20261007/s2_original_port_spectra_preserved.zip `
  --old-watch-dir C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/recover_te_s2_20261006/watch/recover_te_s2 `
  --output NEW_existing_pins.json
```

The package already contains actual_existing_pins.json made from those
existing inputs, before any new first-wave result. Preserve that file and its
reported SHA. It pins the old archive, task09/10 checkpoint bytes/semantic
identities/source maps and base/task09/task10 member NPZs. The archive contains
the ORIGINAL shift30+/-2nm spectra. No values are invented or obtained from a
new solve. The historical center is a read-only export; the endpoints have
separate recovery-generated source roles. Whole-code equality with current
sources and complete original-FSP creation provenance are not asserted.

When actual four checkpoint/NPZ pairs have been fetched, run locally with
exclusive NEW compact and full-spectral outputs:

```powershell
python -B analyze_firstwave.py `
  --repo C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes `
  --evidence-dir C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/docs/readiness_evidence_20261007 `
  --pins actual_existing_pins.json --pins-sha256 EXACT_RECORDED_PINS_SHA `
  --output NEW_firstwave_review.json --spectra-output NEW_firstwave_spectral_differences.json
```

The compact report compares both Tout and modal peak3: corr30 new and prior
half-pairs' central derivatives/even terms, both endpoint scalar and spectral
differences, and spectral derivative differences. Shift30 h0.5 is compared
with ORIGINAL h2 when accessible, preserving both the old native center and
the exact repeated center. Each pair records peak indices and exact center
peak3-Jacobian projection. Full spectral derivative arrays are saved separately.
Production peak selection uses the frozen engine's Bragg-resonance selector,
with dead/off-window checks; it does not substitute a global passband maximum.
Output records actual new checkpoint/NPZ SHA256s for lead review.

Omit BOTH optional --old-spectra/--old-watch-dir when preparing pins to get an
explicit unavailable old-shift comparison. Missing pinned old files later are
also marked unavailable; suspicious existing files with changed hashes fail.
Required new first-wave files never become an optional comparison. Missing
or tampered required inputs fail without publishing a measured result.

The report always says REVIEW_REQUIRED_CALIBRATION_FAILED_UNCHANGED and
no_automatic_later_approval=true. Two corr30 pairs are observed repeat
differences, not a statistical noise bound. Two shift30 step sizes establish
neither asymptotic order nor an exact infinitesimal gradient. The parent must
review measured evidence before choosing whether to publish the separate
later-stage marker. This tool never writes that marker.

Offline tests:

```powershell
python -B test_firstwave.py
```

Six groups pass: complete synthetic new-wave fixture with actual trusted old
inputs; task/sign/point/spec/source/marker tampering; altered grid even after
NPZ digest update; scalar/missing required inputs; optional unavailable old
comparison and archive-hash tampering; analytic odd/even and exact peak
Jacobian projection. Synthetic new spectra are explicitly fixtures, not170962
results. Fixtures and outputs remain in this package; no original is modified.

## Integrated checkpoint

The arithmetic/analyzer sources are unchanged. Tests resolve the repository from their file location and preserve fixtures in TEMP. Pins were regenerated against the byte-identical repository-preserved archive, avoiding a TEMP archive dependency. Integrated six tests pass; no actual firstwave report exists.

Exact integrated pins SHA256: `cff7165a1cc6e7c4aae9024b6ce3dda8d1c9d62dfc007040bd0deaf6f14c7581`.
