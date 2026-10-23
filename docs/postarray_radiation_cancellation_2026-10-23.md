# Radiation cancellation on the TE overshoot by designed SiN post arrays (2026-10-22/23)

**Result (MEASURED, production mesh, short-run method):** an array of 238 full-height SiN posts per side, placed by a Born dipole model
beside both arms of the TE overshoot pi-shift grating (Nt80x), raises the intrinsic Q from **Q_i 5.35e5 to 8.74e5 (+63 %)** with the
spatial mode width held (16.15 -> 16.11 um). Peak T at the same length (96 periods/side) rises 0.664 -> 0.770. Derived Q at -3 dB
(0.293 Q_i): **1.57e5 -> 2.56e5**. Independently recomputed from the raw files with a separate script: ratio 1.633 (fit/peak-pick
variants move it < 1e-5). Open: converged-mesh check (Athena 269410), direct -3 dB device, GPT adversarial review of the result.

This is far-field (all-harmonic) radiation cancellation: the posts are secondary radiators whose fields interfere destructively with the
grating's leak over the whole sphere. Measured far-field power per stored energy fell to 0.62 of the control (model predicted 0.69).

## Device and constraints
- Base: TE overshoot Nt80x (Itai), pitch 514 nm, 96 periods/side, cavity 950.3 nm, lambda 1619.5 nm, SiN n 1.97 / oxide n 1.444, 350 nm.
  Widest teeth 1800 nm (sidewall |y| = 0.9 um at x ~10-30 um on both arms).
- Fab rules (user): planar only, same SiN layer, full 350 nm height, oxide elsewhere; nothing reachable by apodization / tooth widths /
  tooth shifts / cavity changes counts. Spec = Q at -3 dB at fixed mode width.
- Posts: cylinders r 50-100 nm (pa3/pa2) or 50-120 nm (pa4/pa5), mirrored at +-y, edge gap >= 100 nm within a row, rows 300 nm apart,
  first row y = 1.2 um (>= 0.2 um from the widest tooth). Posts sit OUTSIDE the fine mesh override (|y| <= 1.08 um) on the 50 nm
  background mesh (conformal variant 0); the dy29 variant refines only the core, not the posts.

## Results (all MEASURED; control = stored Nt80x T340, Q_i 5.35e5; scored with python_tools/radcancel/solve_v4.py --short --energy)
| Device | Athena job | Posts / side | Q_i | dQ_i | T | width (um) | far-field P/Pc (model) |
|---|---|---|---|---|---|---|---|
| single comb +12 um, 51 deg, r63 | 267628 | 31 | 5.75e5 | +7.5 % | 0.682 | 16.15 | 0.980 (0.975) |
| same at dy 29 (vs dy29 ctrl 5.67e5) | 268446 | 31 | 6.14e5 | +8.3 % | 0.690 | 16.13 | 0.929 |
| left comb -12 um circle 0/120/240 | 268443 | 31 | 4.52/5.73/5.00e5 | -15.5/+7.1/-6.5 % | | | fit optimum 147 deg |
| pair R51 + L147 (independent phases) | 268986 | 62 | 6.25e5 | +16.8 % | 0.701 | 16.15 | 0.856 (0.958) |
| one-line comb, period 539 nm, centre empty | 269305_0 | 67 | 6.14e5 | +14.8 % | 0.697 | 16.16 | 0.842 (0.943) |
| pair + 2nd rows (R y1.3, L y1.9) | 269305_1 | 124 | 6.42e5 | +20.0 % | 0.707 | 16.15 | 0.842 (0.940) |
| **post array pa2 (rows 1.5/1.8)** | 269020_1 | 136 | 7.51e5 | **+40 %** | 0.741 | 16.15 | 0.797 (0.813) |
| **post array pa3 (rows 1.2/1.5/1.8)** | 269020_0 | 238 | **8.74e5** | **+63 %** | 0.770 | 16.11 | 0.620 (0.693) |
| 2nd row y1.9/u0.85 on the single comb (dy29) | 268671 | 60 | 5.87-5.98e5 | below single at all phases | | | dead |

Negative/closed this round: in-plane SiN mirror strips, wider cores, a second comb row at y 1.9 um, same-phase comb pair (-1 %).

## Method
1. **Two-lobe physics of one comb** (measured): a periodic comb radiates two mirror lobes (u_x = +-0.96) from the cavity standing wave
   whose phases rotate oppositely with comb shift, so one comb (one phase) cannot cancel both; the two arms are not mirror images
   (right arm starts narrow, left starts wide) and need independent phases (51 vs 147 deg).
2. **Born dipole model** (`python_tools/radcancel/postarray_born.py`): drive = the control's complex E on the far-field tube side plane
   (y 2.104 um) back-propagated by angular spectrum to the post plane (valid for y > 0.9 um only: pure oxide between); post = point
   dipole with cylinder polarisability; far field = free-oxide dipole radiation, same energy normalisation as the control far field C.
   No CMT. The model has no fitted parameter (the design step uses an empirical response scale 0.70): |B|^2 within ~10 % on 8 FDTD comb rows, pattern overlap ~0.6, direction right in 6/6 checks; it
   under-predicted single-comb/pair gains ~3x but was close on the arrays (0.693 vs 0.620, 0.813 vs 0.797).
3. **Design** (`postarray_ceiling.py`, `postarray_design.py`): response matrix on a 50 nm site grid per row; post amplitude a = (r/63)^2
   >= 0 makes min |C + sum a_n f_n|^2 convex (NNLS ceiling), then discrete posts (free x, radius bounds, spacing penalty, L-BFGS-B,
   gradients FD-checked, measured scale 0.70). Ceiling for rows y 1.3-1.9: 33 % of the leak (positivity-limited; signed amplitudes
   would allow 48-66 % but need sub-oxide index = air voids, excluded).

## Uncertainty (DERIVED) and GPT adversarial review (session 01a16842)
GPT reproduced +63.3 %; a common 0.008 T bias leaves +62 %; erasing the gain needs T 0.64 vs 0.77. OPEN: (1) post-region mesh (posts on the
50 nm background; decisive test = matched control/pa3 with a ~25 nm override over the posts); (2) Q_i = Q_L/(1-sqrt T) assumes symmetric
port coupling (left/right arrays differ; a direct -3 dB device settles it); (3) P/Pc is radiation per a z=0 core-field proxy (monitor
|y| <= 0.94 um), not total stored energy; with radiation share 0.8-0.9 it implies only +44..+52 %, so the loss budget does not close yet.
Short-run T correction +-0.01 -> about +-6 points on +63 %; port-calibration bias < 1 point (same method both); mesh unknown (single comb
moved +7.5 -> +8.3 % at dy 29); Q_3dB = 0.293 Q_i assumes Q_i holds when periods are added to reach T = 0.5. Best estimate +55..+70 %.

## Running / next (2026-10-23)
- Athena 269408: pa4_r120_x30 (rows 1.2-2.1, |x| <= 30 um, 319 posts/side, model 0.654) and pa5_r120_x30 (rows 1.2-2.4, 379/side, model
  0.646). Far-field P/Pc INVALID for these (posts at/outside the tube side face y 2.104 um); Q_i/T/width valid.
- Athena 269410: pa3 at dy 29 vs the stored dy29 control (267605_0, Q_i 5.67e5).
- Next: a direct -3 dB device (more periods; coupling Q 1.22e5 -> ~3.6e5) of the best array; Q at -3 dB then measured directly as Q_L.

## Files
- Designs (post lists): `runners/sweeps/postarray_overshoot_designs.json` (pa3_r100, pa2_r100, pa4_r120_x30, pa5_r120_x30).
- Runners: `runners/sweeps/farfield_sph_15um_postarray_overshoot.py` (+ `_big`, `_dy29`), `farfield_sph_15um_comb_overshoot.py`
  (+ `_dy29`, `_2row`, `_v3`).
- Figures: `matlab_plotting/studies/plot_postarray_overshoot.m`, `plot_comb_overshoot.m` ->
  `results_from_athena/comb_overshoot/devices_*.png`, `comb_devices.png`.
- Results (keep, not in git): `results_from_athena/comb_overshoot/*.mat`; model work files (control far field, response matrices,
  design npz, scan scripts) in `results_from_athena/comb_overshoot/postarray_work/`.
- Reproduce a design: `cd python_tools/radcancel; TAG=_r4 ROWS_UM=1.2,1.5,1.8 KCUT_PER_UM=7.5 python postarray_ceiling.py;`
  `TAG=_r4 R_MAX=100 python postarray_design.py` (writes `postarray_work/design*.npz`; P/Pc printed).
- Log: memory `project_radcancel_overnight_2026-10-22.md`, `project_radcancel_overnight_2026-10-23.md`; GPT sessions in `docs/ASK_GPT_BRIEF.md`.
