---
name: highq-short-run
description: Measure a HIGH-Q pi-shift grating (Q_L ≳ 2e4) with a SHORT FDTD run (~2 amplitude decay times instead of ~12) — Q_L from the time-trace decay, far-field harmonic table, corrected radiated power and peak T, loss budget. Use before dispatching ANY device whose expected Q_L makes a full ring-down cost hours (overshoot apodizations, long devices, Q3dB-class lengths), or when asked how long a high-Q run must be.
---

# highq-short-run — validated 2026-10-20 (user: "critical, those simulations take a very long time")

## Why
A steady-state run needs ~12 amplitude decay times τ = 2Q_L/ω. At Q_L 1e5 (1.6 µm) τ ≈ 170 ps → ~2 ns ≈ 6-7 GPU-h
per device (measured speed: 340 ps = 67 min on H200, N=96 overshoot + far-field tube). Most quantities do NOT need it.

## What a run of ~2 τ gives (GATE B2, Athena 265023: stored 20-um overshoot, 19/26 ps vs its full 150-ps run)
| quantity | how | accuracy |
|---|---|---|
| Q_L | fit of ln\|envelope\| of the `time_cavity` trace (NOT the spectrum) | exact (7736 vs 7736) |
| λ_res | frequency of the same trace / port finder | ≤ 0.01 nm |
| harmonic table, mode width | shape of the mode | 0.01-0.03 pt |
| radiated power (tube) | source-normalized × γ², γ = 1/(1−exp(−ω T_run / 2Q_L)) (Ansys high-Q correction) | 0.4 % |
| peak T | port T × γ² | ~0.006 LOW at 2 τ (0.012 at 1.5 τ) → quote ±0.01 |
| Q_i, Q_c | t=√T: Q_i = Q_L/(1−t), Q_c = Q_L/t (symmetric two-port) | ~±2 % |
NOT usable from a short run: the spectral FWHM / T(λ) shape (truncation-broadened), R, and any tube/end-cap flux
RATIO (fields are the driven response, not a pure decaying mode).

## ★Never use monitor time apodization ("start recording at 40 ps")
The GPU solver silently resets it to 'none' (Ansys GPU limitations; MEASURED 265002: windows at 30 and 45 ps gave
identical fields). The knob was removed from the engine 2026-10-20. Not needed anyway (table above).

## Recipe
1. **Locator** (only if λ is unknown to better than ~0.1 nm): same scene, 60 ps, wide window (e.g. ±10 nm).
   Q_L and λ from the trace (fit from ≥30 ps), cross-check the port peak finder.
2. **Production**: `BASE.farfield.short_run = True` (end caps + time traces; run length in the file tag),
   `TM_SIM_TIME_PS` = 2·(2Q_L/ω) from step 1, source window 4 nm (short pulse, as in the gate), far-field monitors
   in their OWN window ±0.25 nm around λ (`farfield.use_2d_window=True`, `monitors.monitor_2d_center_nm/span_nm`)
   so the 81 projection points sit ≤ 0.2 linewidth from resonance. Tube x-span must exceed the apodized region.
   Template: `runners/sweeps/farfield_sph_15um.py` ROUND C_* (+ thin modules `_c_nt80` / `_c_hh60`).
3. **Analysis**: `python python_tools/ringdown_budget.py <mat>` (Q_L, γ², T, loss, Q_i, Q_c, Q_rad, width), then
   `python python_tools/farfield_multipole.py <mat> --lmax 150 --csv` (harmonic table; lmax 60 for plain gratings).
   Plot: `matlab_plotting/studies/plot_farfield_sph_15um.m`.

## Cost reference (MEASURED)
Locator 60 ps: 14 min. Production 265 / 340 ps: 53 / 67 min (H200). Same devices at full ring-down: ~6-7 h each.

## When NOT to use
Low-Q devices (Q_L ≲ 1e4: a full run is already ≤ 150 ps) and any number that needs T at the 1e-3 level (one full
run as an anchor). Related memory: project_highq_measurement_adequacy.md, project_farfield_sph_15um.md.

## ★Validation 2026-10-25 (post-array program, overshoot Nt80x, Q_L ~1.1e5, 340 ps)
- Truncation correction must start at the EXCITATION (cavity envelope peak, ~0.4 ps here), not t_end - 5 ps: the old form over-corrected
  peak T by ~1 % relative (Q_i gains ~3 % high). Fixed in python_tools/radcancel/solve_v4.py (commit 4ced19e).
- Q_L from the cavity decay: refits over end 200-340 ps / start 15-100 ps agree to 1e-6 (GPT audit).
- Coupling symmetry check (free, every device): late-time power ratio of trace_time_input vs trace_time_output = gamma_L/gamma_R;
  measured 0.987-1.007 on 13 devices -> eta >= 0.99996, so Q_i = Q_L/(1-sqrt T) holds. Re-check it for any design that is strongly
  asymmetric between the arms.
