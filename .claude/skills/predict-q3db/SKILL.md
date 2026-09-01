---
name: predict-q3db
description: Predict long pi-shift-grating observables (T, lambda, Q_L, spectral+spatial FWHM) and design Q3dB-style devices (any dB point, any mode width) from stored calibration — one confirmation run instead of a tuning ladder. Use when the user gives a new device result and asks what a longer/shorter device gives, where the -3 dB (or other) crossing is, what corrugation hits a width target, or asks to refit/validate the q3db predictive engine.
---

# predict-q3db — the q3db predictive engine

★**Read `python_tools/Q3DB_PREDICTOR_HANDOFF.md` first** — self-contained state
(model, backtests, the three live validations, rules, parked list). Running
state in memory `project_q3db_predictive_engine.md`. Program authorized to use
CMT (user 2026-08-31); the CMT ban remains ONLY inside the lumopt2
optimizer/width-wall.

## The three tools (python_tools/)

- `bragg_cmt.py` — piecewise Erdogan CMT/TMM engine (kappa(z) apodization,
  pi/fractional plates, z-dependent loss, envelopes). `python bragg_cmt.py`
  runs its gate suite — do that after ANY edit to it.
- `calibrate_q3db.py` — loads STORED results only, fits per-family parameters,
  runs the hold-out backtest matrix B1-B13, writes `q3db_calibration.csv`.
  `python python_tools/calibrate_q3db.py` from the repo root IS the
  verification; rerun after any new result lands or any model change.
- `predict_q3db.py` — edit the knobs at the top, run. Modes: observe / design /
  extend (anchor on ONE new measured row, borrow family shape).

## The workflow for "here is a new result, extend it"

1. Get the row: pol, corr, pitch, N, T_peak, Q_L (= lambda/|spectral_fwhm|),
   lambda, spatial width, and WHICH MESHER/pipeline (conformal q3db family vs
   PVA optimizer frame — never mix; the tool warns).
2. Set `MODE="extend"`, fill `ROW`, pick `BASE_FAMILY` with the SAME
   polarization (families are single-pol by name: tm_*/te_*/itai_*).
3. Read the printed validity lines — they are rules, not decoration:
   - anchor/calibration device must have 2*kappa*L >= ~3.2 (c325: N >= ~93);
   - T +-0.03 trusted to ~30 periods beyond the anchored range, band by ~45;
   - single-row anchor: Q_L good to ~8-15%; a second row ~30 periods away
     pins it to ~3% (walk-forward: +14.2% -> +6.7% -> +2.3% as rungs 2->3->4).
4. Any-dB target via `TARGET_DB`; width target via `TARGET_WIDTH_UM` (corr
   knob: per-pol measured 1/width-vs-corr line; corr rescaling of Q_i uses the
   TM-measured corr^-2.9 — EXPECTED-grade for TE).
5. The tool prints the ONE confirmation-run spec with pre-registered pass
   bands. Dispatch that run (add-study + dispatch-study skills), compare, then
   rerun `calibrate_q3db.py` so the new row joins the calibration.

## Standing model rules (violations caused every historical failure)

- Extrapolate Q_c, NEVER ln T (measured: lnT-linear missed the crossing +191%).
- Q_i needs the SATURATING fit; a pure power law through the knee gives
  garbage exponents. Q_i is the failure surface — distrust it below/at onset.
- kappa is linear in corr (coherent channel, 0.1-1.3% over 276-400);
  Q_i(corr) is a SEPARATE radiative law (~corr^-2.9 at fixed N, TM).
- Never calibrate kappa on a Q level (ill-conditioned, A = sqrt(T)/(2(1-sqrt(T)));
  use widths (box-independent) or the Qc GROWTH between two rows.
- The CMT engine is the fixed-N SHAPE tool (apodized widths <1%, spectra,
  Qc shape-ratio anchored on a measured row). N-trends of width and Qc go
  through the empirical fits. Light-cone leak ranks envelopes in the Qi
  GROWTH phase only — it does not see the saturation ceiling (2026-09-01).
- Pre-register every prediction (bands in the runner docstring) BEFORE the
  confirmation run; compare after; record hit/miss in the memory file.
- ★A KNOB TRANSFORM NEEDS BOTH TERMS (learned 2026-09-01, corr knob rung 0).
  Moving a family to a new corrugation changes Qc's RATE (kappa prop. corr)
  AND its LEVEL. Applying the rate alone put Qc +31% off at corr 448 (T
  missed the band by +0.087 while the WIDTH knob was right to 2.1% and Qi's
  corr^-2.9 to 3.8%). The intercept term (-0.002818 per nm, fitted on the
  STORED N=150 corr ladder, residuals <=3.5%) fixes it to -7.6%.
  GENERAL RULE: before trusting any knob that moves a family, check the
  transform reproduces the STORED ladder in that knob at fixed N — that
  check was available for free and would have caught this pre-dispatch.
- ★DECOMPOSE EVERY MISS INTO Qc AND Qi BEFORE TOUCHING THE MODEL. The rung-0
  T miss looked like a broken knob; the decomposition showed Qi (the risky,
  radiative half) was RIGHT to 3.8% and the error was entirely in Qc (the
  coherent, cheap-to-fix half). One measured row + the two-term fix then
  designed the confirm rung to T -0.002 / Q_L -0.5%.
- One anchored row makes a NEW-corrugation family design-grade in one more
  run: c448 rung 0 (measured) -> rung 1 predicted T 0.512 vs 0.5097 measured,
  Q_L 4666 vs 4644, width 14.1 vs 14.19. That is the standard recipe for any
  device family the calibration has never seen.
