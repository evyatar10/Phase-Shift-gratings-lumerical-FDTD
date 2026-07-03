---
name: fetch-results
description: Download finished FDTD results from Athena into results_from_athena/, render the study's MATLAB plot headlessly, and reply with full local file paths. Use when the user asks to download results, get/see a plot or figure from a finished run, or says "give me the graph/image of ...".
---

# fetch-results

The standard post-run pipeline, executed the same way every time (this exact sequence
was hand-assembled in essentially every results session):

## 1. Download

Preferred (handles paths + skips `.fsp`):
```bash
bash athena/deploy_athena.sh --results-no-fsp
```
For a single study or ad-hoc files, targeted scp is fine (plain host-first form, never
env-prefixed):
```bash
mkdir -p results_from_athena/<study>/results
scp "evyatarrubin@athena.technion.ac.il:~/bragg_sim_athena/results/<study>/results/result_*.mat" results_from_athena/<study>/results/
```
Gotcha from history: a killed/partial download leaves a stale truncated `.mat` that
loads garbage — if a file loads oddly, re-download it before debugging physics.

## 2. Sanity-check before plotting

Apply the check-result conventions on at least one file (stored
`resonance_wavelength_nm` in-window, peak T above the dead-device floor ≈0.0008,
Q = λ/|spectral_fwhm_nm|). If the check fails, lead with that — don't hand the user a
plot of a dead device.

## 3. Plot (headless MATLAB)

```powershell
& "C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "cd('c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\matlab_plotting'); <plot_script>"
```
- Pick the existing script for the study (`plot_transmission.m`, `plot_resonance_vs_param.m`,
  `plot_transmission_compare.m`, ...) before writing a new one.
- MATLAB `-batch` is synchronous but slow to start; give it a generous timeout.
- Watch the UTF-8 / underscore-in-title gotchas (`reference_matlab_local_verification.md`):
  use `'Interpreter','none'` for filenames in titles.
- Plot cosmetics the user has corrected before: title should carry the physical
  dimensions + resonance λ + peak T; keep legends compact; don't label plots "zoomed".
- Field-map view naming is deliberately NON-standard in this project: the XZ monitor
  is labeled **"Top view"**, the XY monitor **"Side view"**; x (propagation) is always
  the horizontal axis (z vertical for XZ, y vertical for XY; ux horizontal in far-field).

## 4. Deliver

Move generated `.png`/`.fig` next to the data (`results_from_athena/<study>/results/`)
— never leave them in `matlab_plotting/` and never `git add` them (CLAUDE.md §7).
End the reply with the **full absolute Windows paths** to every figure and the results
folder (e.g. `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\<study>\results\<fig>.png`),
unprompted — the user has had to ask "give me the full link" 21 times.
