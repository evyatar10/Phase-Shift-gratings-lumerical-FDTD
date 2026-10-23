# HANDOFF — radiation cancellation of the pi-shift Bragg grating (2026-10-21)

Written for: the next Claude session (Fable), continuing this program. Self-contained; read this first, then
`docs/radiation_cancellation_overnight_2026-10-20.md` (full chronological log) only when a detail is needed.
Simulation readouts (Athena, Lumerical 2026 R1.3, conformal dx 50 nm unless stated) are MEASURED at their stated numerics; fitted source
locations, response solves, analytic reach estimates and proposed thresholds are DERIVED or PROPOSED. GPT reviewed this document
(`docs/HANDOFF_radiation_cancellation_2026-10-21_GPT_handoff_review.md`); its 15 fixes are applied, including the two tool-path repairs
(harm_matrix_build_cache.py, reach_opt.py).

## 1. Goal (owner's words, meeting 12.10 + this session)
Cancel the radiated leak of the pi-shift Bragg grating with ADDED / extra perturbations that cancel a chosen spherical harmonic,
then several harmonics, by a linear combination of measured responses — "beyond inverse design": changes the adjoint optimizer
can already make (per-tooth width/contrast, tooth shifts, whole-cavity width) are NOT interesting. x2 in Q at -3 dB is a target,
not a limit. Owner's current direction: **work on the PLAIN TE device first (simpler concepts), find its optimal cavity, then add
perturbations on top of that optimum**, and only call a gain new if it is not just a better cavity.

## 2. Devices
| device | definition | control far field (file) | key numbers |
|---|---|---|---|
| plain TE | corr 316, pitch 500, mean W 800 (642/958), cavity 250 nm at 800, N 96, TE | `results_from_athena/farfield_sph_15um/results/result_N96_avg_C316_Ybox6p8_Zbox6p8_AS1em12_ff.mat` (round A, 150-ps steady state) | lambda 1558.5, Q_L 3224, Q_i 2.02e4, mode FWHM 15.29 um |
| plain TM | corr 400, pitch 516.83, W800, N96 | `.../farfield_sph_15um/results/result_N96_TM_avg_Ybox8p0_Zbox8p8_AS1em12_ff.mat` | Q 2634 |
| lab Nt80x | Itai overshoot, pitch 514, N 96, cavity 950.3 | `.../farfield_sph_15um_c_nt80/results/*T340ps_ff.mat` (340-ps short run) | lambda 1619.47, Q_L 9.9e4, Q_i 5.35e5, W 16.15 |
| Nt80x +20 parent | inner contrast +20 nm on periods 0,1 (narrow -10 / wide +10) | `.../farfield_sph_15um_contrast/diff_g0001_p20nm/results/*_ff.mat` | R 0.684, Q_i 7.49e5; dy-37 check R 0.717 (Q_i +34 %) |

## 3. Method (how every number here was made)
- R = radiated far-field power normalised by a STORED-ENERGY PROXY, relative to the stated control: cavity-point DFT for short runs,
  Lorentzian/Q/T proxy for steady runs. Verify the actual integrated stored energy before certifying any incremental gain.
  Q_i is DERIVED as Q_L/(1-sqrt(T)) (coupling-symmetry and port-calibration assumptions, CLAUDE.md 2), not independently measured. Short runs (Nt80x): divide each far field by the DFT of
  that row's cavity time trace (`python_tools/radcancel/solve_v4.py --short`). Steady state (plain): Lorentzian normalisation (no flag).
  Response b = row - control (common phase reference, never phase-aligned per row). Overlap eta^2 = (Re<c,b>)^2/(|c|^2|b|^2).
- Harmonics: `python_tools/farfield_multipole.py` (E with l+m even, M with l+m odd only; |a(l,m)| = |a(l,-m)|).
- Tools: `python_tools/radcancel/` (README there). Long result names break Windows MAX_PATH: copy to short names first.
  `--pair i j` = equal +/- probes indexed from the first non-control file; NOT a full/half check (compute ||b_full - 2 b_half||/||b_full||
  separately). Keep `--lcut` off for acceptance. solve_v4 writes `<control>.alpha_v4.npy` next to the control: run on scratch copies.
  `--slam/--sW` are soft penalties and coefficients default to +-1: the constrained antenna solver of step 5.4 (hard lambda/W bands,
  physical bounds, untargeted-power cap) does NOT exist yet - write it before using any solution. harm_matrix_* caches are built for
  the Nt80x short-run basis; rebuild for the plain parent.
- Every new row gets a property assert in its module's `__main__` (CLAUDE.md 5). Check the layout `*_p0.log` on any fast failure.

## 4. What is established (MEASURED unless marked)
- **Source-basis representability** (vector-current fits, held-out; DERIVED, not a unique location of physical sources or proof of
  controllable emitters): plain TE = first three wall steps on each side of the cavity (x = +-a, +-3a, +-5a, a = pitch/4): 0.99 explained; plain TM = distributed over +-10 um (envelope tail; fits P(u) ~ 1/((beta-ku)^2+kappa^2));
  Nt80x: junction-pair fit 0.39 -> 0.15 and five-steps-per-side fit 0.60 -> 0.44 from control to the +20 parent (compact part reduced).
- **Lab device**: the inner contrast is the only strong reshaping knob (R 0.684, minimum at +20 nm; +25 0.697, +30 0.719) and the adjoint's
  own first step points at it (cos 0.25) -> inside inverse design. Added TABS: periodic A/B R 0.971/0.976 vs the original control;
  off-lattice sets 1.035-1.260 vs the +20 parent; the arms-independent solve PREDICTS 0.994 (not a fabricated combination); the matched
  set's even-sector correlation is 0.08. Aperiodic C-F were never run (held). Cavity-centre tab: null (0.995).
  **7 longitudinal NOTCH sites, mirrored across y = 14 rects (5 nm, cladding index, cut into the left-arm wall at x -330..-4057 nm): R 0.896 vs the +20 parent (Q_i +9 %,
  dlam -0.065, dW -0.002); 2.5 nm 0.934** -> 0.613 of the original control (DERIVED), Q_i +53 %. Depth scan (266892): 7.5 nm 0.917
  (Q_i 8.19e5, dlam -0.092), 10 nm 0.939 (Q_i 7.96e5, dlam -0.118) -> optimum near 5 nm. Single geometry, not depth-linear,
  NOT yet confirmed on the finer mesh (266894 running) -> candidate.
- **No knob in the analysed stored Nt80x basis is harmonic-selective** (max selectivity 0.10; the contrast knob scales the whole l<=5 block together).
- **Plain TE junction shoulders** (100 nm long, both sidewalls, on the smaller side of the cavity steps): right step +125 nm, 10 nm R 0.958
  (Q_i +4.5 %, eta^2 0.34); 5 nm 0.969 (depth linearity residual 0.64 = FAILED); left step 10 nm 1.025 (adds); both 0.981 = sum
  (additivity residual 0.022).
- **Plain TE cavity size**: width 780 0.990, 820 0.988, 850 0.948, 900 0.953 (non-monotonic: noise or oscillation), 950 0.918 (Q_i +10 %, but dlam +0.27 nm, dW -0.21 um), 1000 0.912
  (Q_i +11 %, dlam +0.335, dW -0.273), 1050 0.899 (Q_i 2.28e4 +13 %, dlam +0.395, dW -0.339); 1100 pending; length 270 nm 1.039.
  So the 800-nm cavity is NOT optimal and a wider cavity beats the shoulder -> the shoulder is not yet shown to be new physics.
  TM analogue (July, stored): rect cavity 1050 = loss -30 %, flat 1040-1075; an inner see-saw then added ~1 % by interference.
- **Analytic reach map** (point dipoles, `reach_map.py`): a feature at x0 radiates around l ~ k x0; signed 2-4-site patterns at step sites
  give 76-86 % in one harmonic for PRESCRIBED currents. GPT review: not a fabrication prescription (real features take their amplitude
  and phase from the cavity; no x mirror in the device).
- **Literature**: Johnson et al. APL 78 3388 (2001) cancelled one multipole by sweeping one knob; ~15 of ~155 citing papers extend it
  (mostly PhC slabs); multi-multipole cancellation by a linear combination of measured perturbations not found (likely gap).
  Closest 1-D work: Ding, Liu, Li, Opt. Express 20, 28641 (2012), https://doi.org/10.1364/OE.20.028641 (coherent sum of interface radiation).

## 5. Program from here (GPT critical review adopted; full text: `docs/HANDOFF_radiation_cancellation_2026-10-21_GPT_program_review.md`)
**FIRST THING TO READ: 266993** - the owner's open question is whether shoulders/notches still cut radiation ON the better
(1050-nm) cavity; if they do not, every plain-TE feature gain so far was cavity re-optimisation.
0. **Decide the feasibility bands** (owner): the plain device's lambda / width tolerance and whether pitch retuning is allowed. Under the
   old bands (|dlam| <= 0.25 nm, |dW| <= 0.1 um) the 950-nm cavity is NOT admissible. Give baseline and antenna the same freedom.
1. **Is the shoulder real?** finer-mesh (y_mesh_cells 20 -> plain dy = 642/20 = 32.1 nm, not Nt80x's 37.5) plain control + right
   shoulder 10 nm (2 rows; modules still to prepare); diagnose the failed depth linearity (physics vs meshing vs normalisation) with the
   existing 5/10-nm pair, inspecting complex fields and integrated U. Do NOT build an antenna Jacobian until the matched finer-mesh gain
   exceeds measured uncertainty and the depth failure is resolved.
2. **Best validated cavity in a box** q = (cavity width, cavity length, inner contrast c of the innermost period pair): local quadratic,
   ten model geometries plus one independent validation; reusing three suitable width points leaves seven new model rows plus one
   validation. Specify the q bounds and the exact contrast mapping; measure +-length, +-contrast and three mixed probes; length steps
   smaller than +-20 nm, set from the measured dlam slope; narrow and refit after a failed validation. Report as "best validated cavity
   in this box", not "optimal".
3. **New control is real only if** J_feat* = min over (q, feature) < J_0* = min over q alone, both feasible, comparable search effort,
   with a verified incremental gain beyond uncertainty and geometry absent from the baseline parameterisation.
4. **First designed antenna**: target E(3,+-3) (~20 % of plain power; recompute at the new parent). Measure 4 independent local
   shoulder/notch directions near +-a, +-3a at the new parent; solve target-down with total power minimised and untargeted power capped;
   validate full + half amplitude (>= 6 rows). Use measured addition AND removal directions (a notch is not automatically minus a tab);
   four knobs are a minimum subject to rank and bounds; measure target and untargeted ABSOLUTE normalised powers, not fractions. Pass: selectivity ||P_t b||^2/||b||^2 >= 0.70, target power down >= 50 %, untargeted power
   not up beyond uncertainty, lambda/W held, full/half residual <= 0.20-0.25; then finer mesh.
5. **Stop rule**: at most 6-12 validated local knobs at the optimised parent; stop the TE harmonic-antenna route if the uncertainty-aware
   optimistic bound cannot give >= 10 % total reduction, or if predicted selective combinations repeatedly fail full/half and finer-mesh tests.

## 6. Jobs (Athena) — HISTORICAL snapshot at handoff; refresh with squeue before acting
Read with solve_v4: `--short` for 266894 / 266892 (Nt80x short runs); NO flag for 266913 / 266947 / 266573 (plain steady state).
Record each task's remote path, short local name and control before readout. Dispatch: cavsize uses `--cards=`, cavity /
cavity_dy37 / junction_plain use `--option3 --spec=` (each module's docstring has the exact line).
| job | module (runners/sweeps/) | rows | control | status |
|---|---|---|---|---|
| 266894 | farfield_sph_15um_cavity_dy37 (row 4) | 5-nm notch at dy 37 | local `results_from_athena/farfield_sph_15um_meshcheck_c5/nt80x_ycells20_diff0001_p20nm/results/dy37_p20_ff.mat` | queued (PD) |
| 266892 | farfield_sph_15um_cavity (rows 6-7) | notch 7.5 / 10 nm | +20 parent | DONE, read (section 4) |
| 266913 | farfield_sph_15um_cavsize_plain (cards 4-9) | width 850..1100 | plain control | 850-1050 done and read; 1100 running |
| 266993 | farfield_sph_15um_onopt_plain (rows 0-2) | OWNER'S KEY QUESTION: do features still help on the better cavity? 1050-nm cavity + right shoulder 10 nm; 1050 + left-arm notch set 5 nm; 800 + same notches | rows 0-1 vs cavsize card cavW_1050nm (`results_from_athena/farfield_sph_15um_cavsize_plain/cavW_1050nm_ff.mat`, R 0.899 vs plain control); row 2 vs the plain control | running (A100) |
| 266947 | same, card 3 | length 230 nm (A100 pin; failed twice on the 12-seat node n317) | plain control | pending |
| 266573 | farfield_sph_15um_junction_plain (rows 4-8) | J_L 5 nm, shoulders at -5a,-3a,+3a,+5a 10 nm | plain control | running; J_L_d5 result file present on Athena, not yet read |
| 265566 / 265652 | farfield_sph_15um_subteeth C-F | on-lattice reference tabs | — | HELD references; releasing them needs a decision consistent with the plain-device-first program |
Local results: `results_from_athena/farfield_sph_15um_{cavity,cavity_dy37,cavsize_plain,junction_plain,phase,phase_plain}/`.
Figures: `results_from_athena/farfield_sph_15um_phase/devices_overview.png`, `.../farfield_sph_15um_junction_plain/plain_junction_devices.png`,
`.../farfield_sph_15um_cavity/nt80x_notch_device.png` (script `matlab_plotting/studies/plot_nt80x_phase_devices.m`). Overview page:
https://claude.ai/artifact/GWFTB2inUjMhCBMqYt3QHV (state of the morning of 2026-10-21).

## 7. Traps hit in this program (all now in CLAUDE.md / skills)
- Card field `cavity_length_nm` is IGNORED by `to_device_kwargs`; cavity length comes only from `cavity_neg_detuning_nm` (length = pitch/2 - it).
- Node n317 (188-SM GPU) needs 12 license seats per solve; under pressure it dies at ~20 s with FlexNet -4,132 -> pin `--gpu=a100` (7 seats).
- A consumed exit-on-change watcher must be re-armed every turn (5 h idle incident).
- Report fractions with their complement; state the origin of any parity split (the x-parity decomposition is origin-dependent:
  43/57 about x = 0; about x0 = +0.195 um 89.5 % even / 10.5 % odd for the Nt80x control, 86.7 / 13.3 for the +20 parent); a reviewer
  re-derives decision numbers.
- Name tests in variant code (`endswith`) silently fall through -> assert each variant's distinguishing property.
- Tab responses on the lab device are ~7x stronger than the first-order model; tab basis failed linearity (0.34); per-row noise floor
  estimated response-discrepancy floor |b|^2/|c|^2 ~ 0.012 on Nt80x short runs (may include nonlinearity; not an identical-geometry
  noise measurement; the plain steady-state floor is not measured).
- Owner's temporary rule from 2026-10-20 (keep ~10 license seats free) was "for now"; probe seats from IGUM before any dispatch.
