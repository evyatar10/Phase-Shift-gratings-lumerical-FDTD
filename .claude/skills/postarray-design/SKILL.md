---
name: postarray-design
description: Design, check and launch a Born-model SiN post array (side scatterers) that cancels the radiated leak of a pi-shift Bragg grating without widening the mode (narrowing is fine, user 2026-10-23) — model gate, ceilings, discrete design, calibrated prediction, FDTD runner asserts, scoring. Use for any "posts / comb / scatterer array / radiation cancellation" device, for predicting how many rows or how long an array should be, and before dispatching any post-array FDTD run.
---

# postarray-design — 2026-10-23 (TE overshoot Nt80x: 3-row array Q_i +33 % at dy29; +63 % at the production mesh was partly artefact)

START with `docs/HANDOFF_postarray_2026-10-23.md` (state, running jobs, Fable R1-R6 plan). Step 2's calibrated form is SUSPECT (Fable
review: the kappa*x1 term failed out of regime; plain Born predicted pa4/pa5); refit on converged devices before pre-registering with it.

Doc with all numbers: `docs/postarray_radiation_cancellation_2026-10-23.md`. Code: `python_tools/radcancel/postarray_{born,ceiling,design}.py`.
Work files (control far field, Gram matrices, designs, 13-device dataset): `results_from_athena/comb_overshoot/postarray_work/`.

## ★★WORKFLOW RULE (user 2026-10-25, after days of model -> 3.5 h FDTD -> miss cycles)
Best MEASURED (2026-10-25): x42_r0910 x2.096 / x42_j3 x2.098 / nw_r2pack_ext42 x2.076 (dy29). Model add-ons delivered 0.25-0.9 of prediction
EXCEPT in empty regions (ext42 1.6x). So, in this order, before ANY post-array design dispatch:
1. **Model gate on stored data (zero GPU, minutes):** the model must pass a HELD-OUT-FAMILY test on all measured dy29 devices (families
   knobs / w4+w5 / ell / nw9+r3pack / x20 / ext42 / x42 / clouds): correct sign of dQ, meas/pred dP 0.5-1.5, |err| <= 0.02 of base power.
   No pass = no design from that model (the incumbent stays; add-ons only in EMPTY regions).
2. **Physics/numerics questions on SMALL scenes (minutes), never on the 3.5 h device:** single-post response beside the strip, mesh
   strength vs radius (post_mesh_probe 270533: production mesh r50 0.65 / r100 0.91 of converged), registration, post height (finite-height
   form factor sinc(k u_z h/2) ~0.84 at |u_z|=1 for 350 nm) — measure, then put into the model.
3. **Full-device FDTD = confirmation only:** batch every candidate of a wave together, each with a pre-registered band and a kill line.
Separate NUMERICS from PHYSICS first: at the production mesh posts are distorted (radius- and registration-dependent), so the referee itself
is biased exactly where increments fail (resizes/fills next to posts).

## Physics in one paragraph
Each post (full-height SiN cylinder, same layer as the grating) is a weak dipole driven by the cavity field at its site and re-radiates;
the far field is leak + sum of posts. Phase comes from the post's x position, strength from r^2. Minimising the TOTAL radiated power
over the whole sphere (= all harmonics at once, Parseval) is convex in a = (r/63 nm)^2 >= 0 on a fixed site grid (NNLS ceiling).
MEASURED on 13 overshoot devices: Q_i gain = 1 / (measured far-field leak) (corr 0.983) -> loss is ~all radiation, so the far-field
leak is the right objective. A single periodic comb radiates TWO mirror lobes (standing wave) and the two arms of an apodized device are
NOT mirror images (one arm starts narrow, the other wide): every arm needs its own phase (same-phase pair -1 %, independent pair +16.8 %).

## ★Converged mesh first (2026-10-23, the false +63 %)
Design against the dy29 control field (`results_from_athena/budget_overshoot/dy29_box6p8.mat`) and judge at dy29. pa3, designed on
the production-mesh leak: +63 % at dy 57.7, **+33 % at dy 28.8** (269410). Production-mesh gains are screening only.

## Recipe
1. **Control** with the far-field tube surfaces (`farfield.save_surface_eh`): short run (overshoot) or steady run (plain) — the model
   picks the normalisation automatically (`postarray_born.SHORT`). Copy it under `results_from_athena/<study>/` (keep).
2. **Gate the model** against measured devices before trusting a design: `python postarray_born.py` (8 comb rows; expect |B|^2 within
   ~10 %, overlap ~0.6). Calibrated model (13 devices): E = C (1 + kappa s x1) + s Bp, s 1.01, kappa 0.30, x1 = Re<C,Bp>/|C|^2
   (`postarray_work/calib_s_kappa.npy`); leave-one-out rms gain error 0.05. Plain Born x0.70 under-predicts gains ~1.6x but ranks right.
3. **Ceilings first (zero GPU)**: `postarray_ceiling.py` (env ROWS_UM, X_MAX_UM, KCUT_PER_UM 7.5, TAG, PA_WORK, CTRL_MAT). Measured
   ceilings on Nt80x: rows to 1.8 / 2.4 / 2.7 um = 33 / 42.5 / 44.4 %; |x| <= 24 / 30 / 45 um = 37 / 40 / 45 % -> in the MODEL length
   beats rows (UNTESTED; TM evidence: posts along the whole grating did not help beyond a few um). Ceilings and optimizer plateaus are
   CONDITIONAL on the model and its constraints, never physical limits (user 2026-10-23) — test one factor per FDTD run; posts within |x| < 4 um matter most. Positivity is the binding limit (signed amplitudes = air voids = not allowed).
4. **Design**: `postarray_design.py` (env TAG, R_MAX, R_MAX_LAST, GAP_NM, ROW_IDX). All-pairs edge gap (any two posts, any rows) is a
   penalty and the save is REFUSED if violated. GAP is a fab number (user: neighbouring-row gap is not physics; 50 nm = one mesh cell).
5. **Predict + pre-register** the calibrated gain for every design BEFORE its FDTD result (memory file of the study).
6. **Runner** (`runners/sweeps/farfield_sph_15um_postarray_*.py` pattern): per-site x/y/r lists from `postarray_overshoot_designs.json`;
   `__main__` asserts: drawn count = design, radius bounds, all-pairs gap, file-tag uniqueness (len, first/last x, first/last y — TRAP
   268992), clearance to the WIDEST tooth (read the per-tooth arrays: Nt80x wide teeth reach 1800 nm, |y| 0.9 um — the avg/corr formula
   is wrong for apodized devices), box edge, and for a new control family: every non-post kwarg identical to the stored control.
7. **Score**: `solve_v4.py [--short] --energy control.mat device.mat`; Q_i = Q_L/(1 - sqrt(T x gamma^2)); sanity per CLAUDE.md 2.

## Validity limits (each cost a correction on 2026-10-23)
- **Drive region**: the field is back-propagated from the tube side plane through pure oxide, so rows must sit OUTSIDE the widest
  tooth (y > 0.9 um on Nt80x, > 0.525 um on the plain 1050-nm cavity). A row inside the teeth gives a fake ceiling (the "50 %" error).
- **Far-field tube**: side face at y ~2.1 um, untapered only to |x| ~35 um. Posts beyond it -> far-field P/Pc INVALID (Q_i, T, width
  still valid). Longer/outer arrays need a larger tube or Q_i-only reading.
- **PML clearance**: keep >= 1.5 um from the outermost post edge (box 6.8 leaves 0.6-0.9 um for rows >= 2.4 um -> use box 8.8, which
  matches the stored dy29 control 267605_1).
- **Post mesh**: posts sit on the 50 nm background (2-4 cells); the core override ends at |y| 1.08 um. The size of a gain can move with
  post meshing (not its sign). Validate only the FINAL design with the matched fine pair (variants `fine` / `finectrl`: dx 25.7 nm +
  25 nm y band; needs SBATCH_MEM <= the 4d_1g cap — 400G is rejected).
- **-3 dB**: Q_3dB = 0.293 Q_i is derived; measure it directly at the -3 dB length (calibration runner
  `farfield_sph_15um_m3db_overshoot.py`: coupling Q vs periods, reusable for any winner).

- **Calibration regime (MEASURED 269408):** the calibrated model holds for r <= 100 nm and edge gaps >= 100 nm (pa2/pa3). Denser /
  larger arrays (r <= 120, 60 nm cross-row gaps, 4-5 rows) measured x1.55 / x1.42 vs pre-registered x1.81 / x1.83 — worse than the
  3-row pa3. Do not extrapolate the model outside its calibration set; change one factor at a time (length vs density) and recalibrate.
- **Far-field window**: the 2D far-field monitors record only +-0.25 nm around the control resonance; arrays that shift lambda by more
  (pa4/pa5 +0.28/+0.31 nm) give invalid P/Pc. Widen `monitor_2d_span_nm` or re-centre for strong arrays (Q_i unaffected).

## Fab tolerance (model, pa3)
Uniform radius +-5 nm: negligible; random r sigma 6 nm: loses ~1/5 of the gain; random x sigma 20 nm: ~1/6; whole-array x shift 50 nm:
large, but posts share the grating's layer and exposure, so overlay is zero.
