---
name: check-result
description: Load a result_*.mat FDTD result and report transmission, resonance wavelength, Q, and spatial mode width correctly — with the in-window / dead-device sanity check. Use when asked to inspect, summarize, or sanity-check a simulation result .mat file.
---

# check-result

Inspect a `result_*.mat` (or any FDTD result `.mat`) and report the standard metrics
the project cares about, applying the conventions from `CLAUDE.md` so the numbers are
right the first time.

## Steps

1. Resolve the file. If the user named one, use it. Otherwise list candidates
   (`result_*.mat` under `results_from_athena/` or the relevant results dir, newest
   first) and ask which one — don't guess.

2. Load it (read-only; do not write anything). In Python:
   `from scipy.io import loadmat; d = loadmat(path)`. Fields of interest:
   `resonance_wavelength_nm`, `spectral_fwhm_nm`, `T`, `wl_nm`, `fwhm_m`.

3. Report, using the project conventions:
   - **Resonance wavelength** = stored `resonance_wavelength_nm`. NEVER `argmax(T)` —
     the global T max sits in the passband, not the defect peak.
   - **Peak transmission** T at the resonance.
   - **Q = resonance_wavelength_nm / |spectral_fwhm_nm|** (`spectral_fwhm_nm` is often
     stored negative — take the absolute value).
   - **Spatial mode width** = `fwhm_m` (energy vs x) — report only if asked about mode
     width / corrugation matching; it is NOT used for Q.

4. **Sanity check before trusting the result** — and say so explicitly:
   - Is `resonance_wavelength_nm` finite and inside the scan window
     (`min(wl_nm) … max(wl_nm)`)? If not → flag "off-window / peak missed".
   - Is peak T above a low floor (a dead device reads T≈0.0008; healthy TM can be ~0.83,
     so use a low floor, not a TE-tuned one)? If not → flag "dead / off-resonance device".
   - If a check fails, **lead with the failure** — do not present derived numbers as if
     the result were valid.

## Keep it minimal

Read-only inspection. Do not create helper scripts or modify the result. A short inline
Python snippet is enough.
