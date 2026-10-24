# Radiation cancellation of the overshoot pi-shift grating — overnight results 2026-10-20

All numbers MEASURED on Athena (Lumerical 2026 R1.3, conformal mesh dx 50 nm, fixed 150 ps, shutoff off, box 6.8/6.81 um,
tube far field) unless marked DERIVED. Control = the 20-um overshoot seed (Itai Nt60, N=98, Athena 182131: T 0.98327,
Q_L 7736, Q_i 9.2e5, radiated 0.70 % of input, W 19.63 um). "Leak" = far-field power per stored energy relative to the control.

## 1. What was asked
Cancel as much as possible of the overshoot device's radiation with an ADDED structure; understand the harmonic content
(E(4,+-4)); test the semi-analytical models against simulation; say what works, what does not, and which optimization is needed.

## 2. The leak itself (DERIVED from the stored far fields, scratch line_source_model.py)
- A point line source on the axis explains only 19 % (20 um) / 29 % (15.5 um) of the far-field power. The full TE symmetry class
  per u_x ring (p_y cos 2n psi + p_x cos (2n+1) psi) explains 99.5 % / 97 %; the wrong-parity class gives 0.000.
- Band-limited source image: 78 % of the 20-um leak within |x| <= 5 um (61 % at 15.5 um), strongest at |x| < 1 um and +-3.9 um.
  The leak is central and compact (the earlier "+-6/13/25 um lobes" statement is superseded).
- 43 % (20 um) / 51 % (15.5 um) of the leak is at grazing |u_x| > 0.8.
- E(4,+-4) at 15.5 um = 4.2 % each (8.4 % together): the largest member of the sectoral chain of a grazing, line-like source.
  Cancelling both perfectly gives Q_rad x1.09 (DERIVED). No single harmonic is worth a design; cancellation is whole-field.
- 15.5-um loss budget (265046/265047, ring-down method): radiation = 90 % (Nt80) / 95 % (HH60) of internal loss, Q_i 5.2e5.
  Ceiling for removing all radiation: x10 at 15.5 um vs x1.7 at 20 um (radiation share 0.42).

## 3. Models vs simulation
| model | prediction | measured | verdict |
|---|---|---|---|
| first-order Born for a 20-nm mean-width RIPPLE (period 4.7 um) | power 4.4x the leak, 46 % per grazing band | 3.3-4.4x, 44-46 % | correct |
| same Born model for the apodization leak | 10x too strong, 88 % grazing | 0.7 % of input, 43 % grazing | fails (cause not established) |
| linear far-field response, direction seed->T9739 | leak 0.875 / 0.834 at alpha 0.25 / 0.49 | 0.897 / 0.862 | good to 3 pts |
| linear 6-knob mean-width solve (lambda, W held) | leak 0.796 / 0.847 at x1 / x0.5 | 0.842 / 0.871 | good to 4-5 pts |
| 2-knob combination (finite-step solve) 0.21 A + 0.40 B | leak ~0.75 | 0.801 | under-predicts |
| naive sum 0.49 A + 1.0 B (routes multiply: 0.862 x 0.842 = 0.726) | 0.726 | 0.711 | routes are independent in effect |

## 4. What works / what does not (20-um device)
- DOES NOT: a long mean-width ripple (20 nm, period 4.7 um, teeth 4-40): radiates 3-4x the leak, overlap eta^2 2.4 %, T 0.983 -> 0.95-0.96.
- DOES NOT: single-harmonic targeting, reach-rule placement, side combs / mirrors / displaced copies (literature + theory: a displaced
  secondary source cancels <= 63 % even at 0.2 um above the core; none published for an apodized cavity).
- WORKS (route A): the inverse-design tooth family. Halfway along the optimizer's own path (alpha 0.49 of seed->T9739) the leak
  is 0.862, Q_i 9.76e5 (+6 % vs seed, +7 % vs the optimizer's endpoint), T 0.98422, lambda +0.03 nm, W +0.01 um. The optimizer overshot.
- WORKS (route B): six +-5-nm mean-width bumps on period pairs {0,1}..{10,11} measured one-sided, one linear solve with lambda and W
  held: leak 0.842, Q_i 9.81e5, T 0.98428, lambda +0.018 nm, W +0.004 um. Single bumps overlap < 1 % (inner) to 15 % (outer); the
  SOLVED combination is the lever. 5 nm is already beyond linear (pair residual 0.36) yet the prediction held to 4-5 pts.
- Routes A and B are correlated (they cancel the same part of the leak): predicted together 25 % vs 21 / 19 % alone.
- BEST DEVICE (combo_sum = 0.49 A + 1.0 B, Athena 265105): leak 0.711 of the seed, T 0.98523, Q_L 7740, Q_i 1.044e6 (+13 % vs seed
  9.2e5), lambda +0.020 nm, W +0.005 um. Routes A and B multiply (independent in effect). Q(-3 dB) scales with Q_i: +13 % at 20 um.
  At 15.5 um (radiation 90 % of loss) the same 29 % radiation cut would be ~+35 % in Q_i (DERIVED); the lab device needs route A's
  direction first (no inverse design exists there yet).

- Headroom (265108): 0.49 A + 1.5 B -> leak 0.811; 0.74 A + 1.5 B -> 0.829. The optimum along these two directions is the
  combo_sum point; the next gain needs a new response basis measured there (Newton round).

## 4b. The harmonic formalism, quantified on the measured 20-um basis (DERIVED, scratch harmonic_jacobian.py)
- Zeroing the K largest harmonics exactly with the 6 mean-width knobs RAISES the total leak: x6.0 (K=1), x19 (K=2), x425 (K=5),
  x426 (K=10), at 35-311 nm amplitudes. The whole-field least squares gives 0.79 (0.80 with lambda, W held) at < 10 nm.
  => "cancel chosen orders" is the wrong objective; minimise the whole far field (the deck's min ||c + B q||^2 form).
- SVD of the real Jacobian: the 6 bumps reach 21 % of the leak (outer group {10,11} 10 %, {8,9} vs inner 7.5 %, common-mode 0.8 %);
  79 % is orthogonal to them. The low-l non-sectoral orders (M(1,0), M(3,0), E(4,+-2), M(4,+-3)) are untouched by mean-width
  bumps; in the best device they fell 75-85 % through route A (the tooth family). A single knob can zero any one harmonic only at
  15-175 nm (non-physical).
- Harmonics remain the right DIAGNOSTIC (which orders a family reaches); the OBJECTIVE is the whole field.

## 4c. Midday 2026-10-20: is the uncancelled half even physical? (CANDIDATE, kill test running)
A Fable physics agent split the control's far field into l<=10 and l>10 (DERIVED from the stored 182131 field): the l<=10 half is
the pi-shift cusp (70 % within |x|<=2 um, 53 % grazing); the l>10 half is a comb of angular lines at |u_x| 0.62/0.72/0.86 from
teeth 6-12 (x-even), only 27 % grazing. A y-mesh staircase model (sidewall edges crossing the ~50-nm override mesh cell every ~5
periods along the apodization ramp) reproduces that comb with correlation 0.82 (0.68 on the 15.5-um Nt80x). The best device
cancelled 54 % of the cusp half and injected an equal amount of NEW l>10 power (24 %) — consistent with knobs that move edges by
0.05-0.25 cell. Kill test: the seed at dy 29 and 38 nm (new inert engine knob MeshConfig.y_mesh_cells; IGUM 113101/113102);
the model predicts the comb envelope moves to |u_x| 0.24-0.40 (dy 29) / 0.48-0.56 (dy 38) and the l>10 share drops; a physical
source pair predicts an invariant comb at 0.60-0.88. UPDATE (held-out complex-field test): moire 0.38 vs physical pair 0.30 at
equal parameters; the 0.82 was mostly the pair fringes both share -> 'weakly favoured, undecided' until the dy rows land. If confirmed, the x2 arithmetic
changes: the physical leak is the cusp, of which 54 % is already cancelled, and every knob response must be re-measured at
a clean mesh. Also: the 15.5-um l>80 band (12 %) is a tube-end artefact (taper test), so its "grazing 60 %" is not quotable yet.

RESULT (dy 37 nm row, Athena 265135, MEASURED): moire KILLED. T 0.98328 / Q_i 9.21e5 unchanged to 1e-5; far power per stored
energy 0.953 of the control; the l>10 comb stays at k_x 4.36/4.83/5.53 /um and its source at +-4 um; only 8.8 % fell in the mesh
model's predicted band (62 %). The l>10 half is physical. Caveat: its fine angular pattern is mesh-sensitive (complex overlap 0.18
between dy 50 and dy 37) while totals are converged (~5 %) -> the dy 29 row decides which mesh is converged; a linear response
basis measured at dy 50 may not transfer. Lesson captured: a §2 noise-floor test is the right first move on any 'half the leak'
claim, and it cost 2 rows.
THREE-MESH RESULT (dy 50/37/29, MEASURED): leak per stored energy 100 / 95 / 78 %; l>10 part 51 / 45 / 33 % (pattern overlap
between meshes <= 0.18); l<=10 part 49 / 50 / 44 % (overlap 0.72-0.84); T 0.98327 / 0.98328 / 0.98464, Q_i 9.21e5 / 9.21e5 / 9.80e5.
CALIBRATED READING (after an independent review): the high-order radiation is substantially UNCONVERGED at dx 50 / dy 50 and its
physical remainder is unknown; nothing here separates numerical from physical by angular order (interference), and the low-order
part is not fully converged either (-11 % from dy 37 to 29). Consequences: every stored leak number is a result at its stated
numerics; the cancellation FOM stays the TOTAL radiated power per stored energy (an l-band objective would reward moving power into
the unresolved band, which is exactly what the best device did with +24 % new high-l power); response bases need a finer-mesh
check before they are trusted; the refinement sequence continues (dy 20 row dispatched, then x and z). The Nt80x's 'radiation = 90 %
of loss' is a dx-50 number; its finer-mesh row (265195) is running.

## 4d. The 15.5-um lab device (Nt80x, N=96): first measured responses (Athena 265111, short-run rows, MEASURED)
Normalisation: each row's far field referenced to the DFT of its own cavity trace at the far-field frequency (phases align to
<= 1.2 deg across rows; this replaced a Lorentzian/S21 scheme that a review showed to be wrong on short runs).
- Mean-width family (+5 nm on both teeth of a period pair, groups {0,1}..{15,17}): 2-10 % of the leak power each, overlaps <= 4 %,
  8-knob solve 4 % with lambda held; the +-pair residual is 0.50 (responses ~ noise). Weak, as pre-registered.
- CONTRAST family (wide +2.5 / narrow -2.5 nm): group {0,1} (the two periods beside the pi shift) ALONE: far power per stored
  energy 0.862, T 0.664 -> 0.696, Q_L 99.1k -> 101.4k, Q_i 5.35e5 -> 6.13e5 (+15 %), lambda -0.002 nm, W -0.004 um,
  overlap eta^2 0.32 (the best single knob measured on any device). Group {2,3}: 0.956, eta^2 5 %. Groups {4,5} 1.066 (worse), {6,7} 0.984, {8,9} 1.010, {10,11} 1.027: the outer contrast
  knobs do nothing; the cancellation lives in the two innermost period pairs (the cusp).
- Bounded two-family solve (lambda, W held) with 14 rows: predicted 0.73 with both inner contrast knobs at the +1 bound,
  i.e. it wants MORE central contrast -> line search dispatched: {0,1} at +10/+15/+20 nm and {0,1}+{2,3} at +10 nm
  (runners/sweeps/farfield_sph_15um_contrast.py, Athena 265257, held until license seats free; A100 rows take ~3.5 h each).
- Incident: the 50-seat license pool was saturated by our own fleet (H200 9 seats/solve, A100 7); the dy-20 row died (FlexNet -4);
  the duplicate H200 gradient lane was cancelled on request and both queued arrays are held/throttled to 2.
- Finer-mesh control of the Nt80x (dy 37.5 nm, Athena 265195, MEASURED): Q_rad 5.87e5 vs 5.86e5 at dy 50, radiated 0.280 vs
  0.275 of input, Q_L 100.4k vs 99.1k, Q_i 5.55e5 vs 5.25e5, lambda -22 pm, W 16.16 um -> on the lab device the total radiation does
  not move with the y mesh; its budget (radiation ~95 % of loss) stands. The matched +5-nm contrast card at dy 37 (265271) is queued.
- Final 15-knob two-family solve on the Nt80x (bounded, lambda and W held): predicted leak 0.718, with the two inner contrast
  knobs at their bound (the line search tells how far they go).
- Also running: the adjoint gradient lane on the Nt80x (244 variables, A100 twin 265174; H200 copy 265121 parked for cancel),
  the Nt80x at dy 37.5 nm (265195, real radiation budget), the 20-um seed at dy 20 (265204), the 20-um Newton basis (265120).

- Contrast line search on the inner pair (Athena 265257, MEASURED): +5 / +10 / +15 nm -> radiation per stored energy 0.862 /
  0.767 / 0.720, Q_i 6.13e5 / 6.79e5 / 7.23e5 (+15 / +27 / +35 %), T 0.696 / 0.720 / 0.733, lambda and width held (|dlam| <= 0.004 nm,
  |dW| <= 0.015 um). Gain per step shrinking; +20 running, +25/+30 queued. This is the re-shaping BASELINE.

- Harmonic narrative of that knob (MEASURED multipole tables, figure results_from_athena/farfield_sph_15um_contrast/
  nt80x_contrast_search.png): the l<=5 block of the control's leak falls 45 % -> 32 / 23 / 17 % at +5/+10/+15 nm (E(4,+-4) 4.2 -> 1.1 %
  each, M(3,0) 4.0 -> 2.0 %, M(2,+-1) 3.0 -> 0.6 %, M(1,0) 2.6 -> 0.8 %) while the total falls to 72 %: the knob cancels the
  low-order cusp harmonics. Independent linear-model prediction for the running rows: optimum near +18 nm, R(+20) 0.72,
  R(+25) 0.76, R(+30) 0.85 (GPT, pre-registered).

- CERTIFIED at the finer mesh (dy 37.5 nm, matched control, Athena 265195/265271, MEASURED): the +5 nm knob gives R = 0.887 vs
  0.862 at dx 50 (|dR| 0.025 <= 0.05), Q_i 5.55e5 -> 6.31e5 (+14 %), Q_L 102.3k, lambda +0.001 nm, W -0.005 um: robust to refinement.
- Harmonic objective settled on this basis (DERIVED): the whole-field solve already cuts the l<=4 block 31 % while cutting the
  total 28 %; buying more block costs total power (5 pts of total for 3 pts of block). Linear predictions from the +5 column hold
  to +10 nm (field error 5 %) and degrade at +15 nm (11 %).

## 4e. The user's family: a SECOND set of small teeth on the sidewalls (added geometry, not re-shaping)
Realised with the scatterer-array machinery: rect SiN sub-teeth 120 nm long x 60 nm deep (full core height) attached to the local
sidewall, mirrored +-y, both arms, 7 per side from |x| = 1.0 um in steps of L2. Physics: a sub-corrugation of period L2 scatters
the +-beta guided waves into u_x = -+(n_eff - lam/L2)/n_clad; L2 = 578 nm -> the grazing lobes (|u_x| 0.85, ~50 % of the Nt80x
leak), L2 = 1000 nm -> broadside; offset dx = phase, depth = amplitude, extent = angular width; no 2*beta component, so the Bragg
coupling is untouched at first order; co-located with the leak, so whole-sphere cancellation is possible (unlike cladding teeth).
First test (module farfield_sph_15um_subteeth.py, 4 rows: L2 578 at dx 0 / L2/4 / L2/2, and L2 1000): the response b(dx) per
stored energy vs the control, the +-pair linearity, and the two-quadrature overlap with the leak. Submitted held (seat rule).

## 4f. Evening 2026-10-20: first added-geometry rows (MEASURED, Athena 265555) and the baseline line search
- Re-shaping baseline (inner-pair contrast, lambda/W held): +5/+10/+15/+20 nm -> radiation 0.862/0.767/0.720/0.684 of control,
  Q_i +15/+27/+35/+40 %. Certified at dy 37 (R 0.887 vs 0.862). Harmonic table: the l<=5 block falls 45 -> 32/23/17 %. +25/+30 running.
- Added geometry, periodic 578-nm tab array (120 x 60 x 350 nm SiN tabs on the wide-half walls, 7/arm): 5 nm protrusion ->
  radiation 0.971, Q_i +1.6 %, lambda +0.107 nm; 2.5 nm -> 0.976, +2.0 %, +0.053 nm. Element linear (|b|^2 x3.8, dlam x2.0).
  Overlap with the leak 3 %: the periodic array is NOT an antenna matched to the cusp -- exactly what the first-order model
  pre-registered (eta^2 ~ 0; element strength within its spread, dlam +0.10 vs +0.107 measured; its PATTERN was wrong).
- Next (queued behind the gradient job, start ~03:00): the aperiodic tab designs C/D (model-designed on the leak's harmonic vector,
  predicted 0.954/0.974) and E/F (arms independent, 12 sites, predicted 0.860/0.921). These are the antenna test proper.
- Objective: whole-field power; the harmonic narrative is read from the tables (l<=4 block -31 % at the total-power optimum).

## 5. Jobs (all Athena, 200G, ~35 min/row at 20 um)
265078/079/080 ripple phi 0/180/90 | 265089 bump basis (8 cards) | 265098 dirval (2) | 265100 zero-profile control (2, incident:
pasted profile was zeros -> rule added to CLAUDE.md S5; the rows proved the cards path = SweepSpec path exactly) | 265103 bump
validation (2) | 265105 combination (2) | 265108 headroom (2). Other chat: 265046/265047 = 15.5-um production rows.
Modules: runners/sweeps/farfield_sph_20um_{ripple,ripple180,ripple90,bumps,bumps_val,dirval,combo}.py, farfield_sph_15um_{ripple*,bumps}.py
(15-um ones NOT dispatched). Scratch analysis (session scratchpad, not in repo): ripple_response.py, bump_solve.py, line_source_model.py,
born_model.py, knob_harmonics.py, ux_marginal.py.

## 6. Recommended optimization
1. Treat the far-field tube power per stored energy as the FOM (with lambda and W as constraints) and run the measured-response
   linear solve on the inverse-design family (tooth shifts + corrugation + average width, 2 runs per knob) -- the mean-width-only
   basis is weaker but cheap. The adjoint campaign already produces this family's direction; re-run it at fixed-time conformal
   numerics with a radiation-aware FOM, and line-search along its path (the midpoint beat the endpoint).
2. On the 15.5-um Nt80 lab device (radiation 90 % of loss): first obtain one design direction (inverse design at N96 / pitch 514,
   short-run method, ~67 min per solve), then the same alpha line search. A mean-width-bump basis there would need ~30 nm bumps
   (leak 40x larger in absolute terms) = nonlinear; not recommended.
3. Keep the per-u_x-ring symmetry-class fit and the band-limited source image as the standard diagnostics (promote from scratch).

## 7. Parked for the user
- Deleting the superseded 15.5-um ripple modules (farfield_sph_15um_ripple*.py, uploaded as source only) and the zero-profile
  job's result dirs (results/farfield_sph_20um_bumps_val/solved_x1, solved_x0.5 on Athena).
- Committing tonight's modules, skill edits (athena-preflight, check-result, dispatch-study), CLAUDE.md S5 rule, docs/ASK_GPT_BRIEF.md log.
- Whether to open an inverse-design campaign on the 15.5-um Nt80 (multi-day GPU).

- 20:50 contrast ladder CLOSED (265649_4 / 265648_5, cavity-trace normalisation vs 265046): +25 nm R 0.697 (Q_i 7.50e5, dlam +0.012, dW -0.026),
  +30 nm R 0.719 (Q_i 7.22e5, dlam +0.015, dW -0.033). Minimum at +20 nm (R 0.684, Q_i +40 %). GPT's linear model put the turn at
  +18 nm but predicted 0.76/0.85 at +25/+30: the curve past the optimum is flatter than linear. Joint 3-row solve: no further gain.
- 20:40 rethink (GPT 01a15a66 + literature agent): sub-teeth A-F were snapped onto the 514-nm lattice = tooth reshaping; leak x-parity
  43 % even / 57 % odd, 76 % toward +x (DERIVED from 265046 far field, lmax 120); mid-gap nodes sit on the tooth STEPS (Koks & van
  Exter 2021); proposal = off-lattice irregular tab set at registrations 0 / +257 nm, 5 / 2.5 nm, on the +20 nm parent (4 rows),
  replacing C-F. Awaiting user decision.

- 22:00 PLAIN-DEVICE PHASE TEST (265799 rows 0/1, MEASURED vs stored round-A control, Lorentzian norm): off-lattice chirped
  10-nm tabs radiate 1.2 % of the plain leak per stored energy at BOTH registrations, overlap eta1 0.0001 (r0) / 0.0096 (r0+half),
  P/Pc 1.010 / 0.991, dlam +0.21 nm, dW +0.02 um. The basis fails at 10 nm on the plain device. Why (DERIVED, lmax 60): the plain
  15.5-um leak is a compact LOW-ORDER source (96 % within l<=5: M(1,0) 13 %, E(3,+-3) 2x10 %, E(2,+-2), E(4,+-4); 28 % grazing),
  while the tab arrays radiate broad HIGH-order patterns (only 30-36 % within l<=5, 42 % above l=10). An extended array cannot
  mimic a cusp-sized source; cancelling the plain cusp would need a feature AT the cavity segment. The Nt80x leak is different
  (46 % above l=10, 51 % grazing), so its rows (265802) remain the real test.
- 22:30 PRE-REGISTRATION of the Nt80x phase rows 265802 (first-order model scratch subteeth_model.py, vs the CONTROL leak; the parent
  is the +20 device so ratios vs the parent are ~1.46x larger): r0 5 nm |b|^2/|c|^2 0.014, eta^2 0.000, P/Pc 1.012, dlam +0.106,
  kappa -2.7 %; r0+half 5 nm: 0.014, eta^2 0.046 but CONSTRUCTIVE (P/Pc 1.066, top E(4,+-4) 4 % each, M(5,0), M(3,0)); r0 2.5 nm 1.002.
  => the model says the half-period registration flips the overlap sign (the phase lever exists) but in the ADDING direction; the
  cancelling registration would be the opposite shift (-half), predicted ~0.96. The model's overlap was 10x low before: read the
  measured eta and sign, then re-solve the registration. Note the 2beta (kappa) term is a tooth-reshaping side effect the model omits.
- 22:45 plain 5-nm row (265799_2): |b|^2/|c|^2 0.0046 (10 nm: 0.0118, ratio 2.6 vs 4 linear), dlam +0.14 (vs +0.21), eta1 0.0003,
  P/Pc 1.0025. Sub-linear in depth on the plain device (or near the floor); verdict unchanged: no overlap at any registration.

## 4g. Night plan 2026-10-21 00:45 -> 10:45 (GPT-verified, session 01a15a66)
- Adjoint it0 (265174, DERIVED from grads_it000 npz): gT 99.9997 % parallel to gLam (high Q); the lambda/W-neutral part is 0.08 % of |gT|;
  its two largest tooth entries are corr of periods 0 and 1 (+); cos with the contrast knob +0.25 => inverse design points at the
  contrast knob (partial alignment, not equivalence). The lane's own trial forward (pred dT +0.064) runs; it may continue to another
  adjoint after an accepted trial (GPT, code-read) -> its 7 seats stay counted.
- 00:50 265802 released (dependency removed, throttle 3): Nt80x off-lattice phase rows r0 5 / r0+half 5 / r0 2.5 nm, A100.
- 00:55 265968: +20 nm contrast at dy 37 (meshcheck_c5 card 1); pass |R_37 - 0.684| <= 0.05 vs the stored dy-37 control = survives y-refinement.
- C-F (265566, 265652) HELD (scontrol hold): reference only, released only if seats and time allow after the phase readout.
- Cavity-local fallback DROPPED: eliminating the whole remaining l<=5 block gives 0.684 x 0.81 = 0.554 at best.
- Phase readout on completion: follow-up (solved set + half depth, 2 rows) only if responses are resolved, the r0 half-depth check
  passes and the bounded nonnegative solve predicts <= 0.85 vs the parent; else "this basis closed"; uncertain -> park.
- Last ordinary start 07:00; final report ~10:45 regardless of unfinished jobs. Seats: ours <= 5 A100 (35/50).
- 01:20 FINAL plan addition (x-parity, user; GPT-checked): 265969 = rows 3-4 of farfield_sph_15um_phase = the r0 and r0+half sets
  on the RIGHT arm only (5 nm, 7 sites, A100, afterany:265802). b_left = b_both - b_right (first order; validate the final combination
  at full + half depth). 4-knob bounded solve (r0 R/L, rh R/L) + x-even/x-odd projection of every response. Follow-up (solved set +
  half depth, 2 rows) only if the constrained prediction <= 0.85 vs the parent exceeds its uncertainty; latest actual start 07:00.
  Offline: the 76 % +x asymmetry on the +20 parent (projector/end-cap consistency, late-time stability) - not via the plain control.
- 01:40 CORRECTION: the "76 % of the leak toward +x" (00:50, section 4g + 01:20 addition) was a BUG (wrong broadcasting of the solid-angle
  weight in the normalising total). Recomputed (taper 15 um): +x share 0.519 control / 0.524 +20 parent / 0.493 plain steady state.
  The 43 % x-even / 57 % x-odd split came from a separate script whose parts sum to 1.000 and stands. Grazing band |u_x|>0.8 moves up to
  3 pts with the tube end-taper (8 -> 25 um) on the Nt80x short runs (plain steady: 0.0) - a small end sensitivity of the grazing part.
- 02:00 FABLE REVIEW (a373bcb3) + direct check: the x-parity split depends on the assumed mirror plane. About x = 0 the leak is 43 %
  even, but about x0 = +0.195 um it is 0.895 even (control) / 0.867 (+20 parent); scan -0.257..+0.3 um in the doc's scratch run.
  The device has no mirror at x = 0 (left arm wide-first, right narrow-first); its effective plane sits near +0.2 um. So the leak is
  ~90 % x-even about its own plane; the odd part is 10-15 %. Consequence: tab sets should be mirrored about x0 ~ +0.195 um, not 0
  (the 0.39-um offset is up to ~2 rad of propagation phase at grazing). Rows 3-4 stay useful as an arms-independent basis; the
  follow-up design (if any) places left tabs at 2 x0 - x_R. Runner fixes: dispatch line (--array-tasks) and N/W label (print only).
  Other review items clean: half_width matches bragg_device, footprints, protrusion, row 3-4 sites, unique tags, parity code, --short norm.
- 03:30 NT80X PHASE ROWS (265802, MEASURED, cavity-trace norm vs the +20 parent; scratch parity_readout.py, parity about x0 = +0.195 um):
  | row | P/P_parent | |b|^2/|c|^2 | b x-odd share | Re<c,b>/|c|^2 | Q_i | dlam |
  | r0 5 nm | 1.131 | 0.148 | 0.91 | -0.008 | 6.74e5 | +0.107 |
  | r0+half 5 nm | 1.260 | 0.108 | 0.76 | +0.076 | 6.14e5 | +0.090 |
  | r0 2.5 nm | 1.056 | 0.048 | 0.83 | +0.004 | 7.18e5 | +0.058 |
  Parent leak 0.867 x-even. Tab sets mirrored about x = 0 radiate mostly x-ODD -> different symmetry sector -> no cancellation (the
  user's symmetry argument, measured). Tabs ~7x stronger than the first-order model (pred 0.021 vs the parent). Linearity 0.339 (gate 0.25).
  Both-arm basis: nonnegative solve gives no gain. Next: parity-matched set (left tabs mirrored about x0), predicted from rows 3-4 as
  b_match ~ 2 even(b_R); dispatch only if Re<c_e, b_R,e> < 0 with a predicted P/P_parent <= 0.90.
- 03:40 DISPATCHED 266004 = rows 5-6 of farfield_sph_15um_phase: PARITY-MATCHED r0 set (right arm unchanged, left tabs mirrored about
  x0 = +0.195 um, clamp moves <= 22.4 nm) at 2.5 and 1.25 nm, A100, afterany:265968 (start ~04:15, done ~06:40). Reviews: GPT (01a15a66)
  - no mirror-based prediction gate, direct test vs the stored unmatched r0 2.5 nm row; Fable (aea838e0) - rows 0-6 geometry, tags and
  the parity split verified (synthetic dipole test), 4/7 pairs on opposite wall types. Own pre-dispatch echo caught a name-test bug
  (rows would have been built mirrored about 0) - fixed before deploy. Pass: total P/P_parent below the unmatched row at the same depth
  AND odd-sector power falls without matching even growth; lambda/W held.
- 09:40 NIGHT RESULTS (MEASURED, cavity-trace norm; INCIDENT: ~5 h idle 03:50-09:10, a consumed watcher was not re-armed -> no follow-up
  before the 07:00 cut-off; rule added to work-alone):
  * Right-arm-only rows 265969: r0_R |b|^2/|c|^2 0.081, Re<c,b> +0.011; rh_R 0.055, +0.010. Derived left knobs: r0_L 0.062 / -0.020,
    rh_L 0.049 / +0.066. Nonnegative 4-knob solve: P/P_parent 0.994 (r0_L at 0.32 of 5 nm). Separating the arms recovers nothing.
  * Parity-matched rows 266004 (left tabs mirrored about x0): 2.5 nm P/Pp 1.067 (unmatched 2.5 nm: 1.056), 1.25 nm 1.035. Matching
    the position mirror does not create a cancelling field.
  * dy-37 +20 nm (265968) vs stored dy-37 control: R 0.717 (dy 50: 0.684), Q_i 7.61e5 vs 5.67e5 (+34 %), dlam +0.004, dW -0.019.
    |dR| 0.033 <= 0.05 -> the contrast gain SURVIVES y-refinement.
  VERDICT: off-lattice sidewall tabs (periodic-chirped, both registrations, arms independent, parity-matched) are closed on the lab
  device at 1.25-5 nm: they radiate 2-15 % of the leak and do not cancel it. Best device remains the +20 nm inner contrast.
- 09:45 matched-row parity split (parity_readout.py, labels positional): unmatched r0 2.5 nm b even share 0.33; MATCHED 2.5 nm 0.71,
  matched 1.25 nm 0.66 -> mirroring the left tabs about x0 steers the tab radiation into the leak's even sector (mechanism works).
  But the even-sector correlation Re<c_e,b_e>/(|c_e||b_e|) = 0.0132/sqrt(0.867*0.0345) = 0.08 with the ADDING sign; even a sign flip
  (a notch) would cancel < 0.2 %. Limit = ANGULAR-PATTERN mismatch of an extended array vs the leak, not parity and not sign.
  Only changes where the leak is generated (inner periods: contrast knob eta^2 0.32) reproduce its pattern.
- 09:50 FABLE RE-DERIVATION (a356037b, independent trace fit/DFT/solves): all night numbers AGREE to 4 digits (dy-37 0.7169 vs 0.6835,
  Q_i 7.61e5 vs 5.67e5; phase rows; 4-knob 0.9938; with matched rows as extra knobs and bounds to 10 nm still 0.9938). New: a per-row
  floor |b|^2/|c|^2 ~ 0.012 (the 1.25-nm row x4 = 0.094 vs 0.048 at 2.5 nm); only NEGATIVE depths (notches 4-14 nm) predict 0.82-0.89,
  a linear extrapolation past a failed linearity gate - flag only. Normalisation is per cavity-point |E|^2; Q_i-implied ratios
  (0.746 dy37 / 0.714 dy50) sit ~4 % above the far-field ratios on both meshes - consistent offset, signs unaffected.
- 09:57 INVERSE-DESIGN TRIAL (265174 eval 1, PVA, 340-ps truncated run, lane's raw numbers): corr period 0 / 1: 0 -> 10.0 / 63.1 -> 70.8 nm
  (the step IS mostly the inner contrast, ~half of +20 nm); FOM 0.4665 -> 0.5056 (+0.039, predicted +0.0645, gain ratio 0.61);
  lane Q_i 2.08e5 -> 2.32e5 (+11 %); dlam +0.0025 nm, dW -0.015 um (pred -0.014). Inverse design independently finds the contrast knob.
  Lane continues (next adjoint since 03:51).
- 12:00 GRADIENT LANE DONE (265174, 26:46 h, exit 0, clean budget stop): best FOM 0.5056 (seed 0.4665). it1 gradient (after the
  accepted step): the lambda/W-neutral part still points at periods 0 and 1 (+0.0019, +0.0010), cos with the contrast knob +0.19
  (it0 +0.25): the optimizer wants MORE inner contrast. State files local: results_from_athena/campaign_te_nt80x_a100/.
  No further iterate dispatched (outside tonight's plan; the measured contrast ladder already shows the minimum at +20 nm).
- 13:10 DISPATCHED 266486 (runners/sweeps/farfield_sph_15um_cavity.py rows 0,1,4,5; A100, 4 concurrent; ~2.4 h): BEYOND-INVERSE-DESIGN
  screen on the +20 parent - centre cavity-segment tab 5 / 2.5 nm, notches at the rh left-arm sites 5 / 2.5 nm (first-order
  hypothesis 0.92 / 0.95, not calibrated). Edge tabs (rows 2,3) held per GPT (01a15a66). Fable (a15d64f2): geometry, notch index
  (mesh order 1, object-index mode), tags, parent identity all clean. Build smoke: 14 notch rects, n 1.444. Gate (GPT): resolved
  >= 10 % reduction vs the parent beyond paired uncertainty, lambda/W held; acceptance only vs the stored dy-37 +20 parent later.
  Figure of all families: results_from_athena/farfield_sph_15um_phase/devices_overview.png.

## 4h. Plain-device semi-analytical models (2026-10-21, user: understand on the plain device first, then transfer the math)
- Envelope-tail model P(u_x) ~ 1/((beta - k u)^2 + kappa^2)^n (+ mirror), kappa = ln2 / mode FWHM, stored plain far fields (farfield_sph_15um
  round A): TM fits with n = 1 (rms 0.35; 33 % in the last grazing bin) -> TM leak = envelope Fourier tail, 1/q^2 (field-jump type).
  TE does NOT fit (rms 1.56): broad, peaks at |u_x| 0.75-0.85 -> a compact source.
- Band-limited source image (scratch line_source_model.py, TE class fit explains 0.999): plain TE leak 90 % within |x| <= 2 um, strongest
  within +-1 um (cavity + first period). The off-lattice tab arrays (|x| 0.6-4.2 um) sat mostly outside the source. (TM image not
  valid with the TE-class columns.)
- GPT (01a15a66 session 21): TE plain = two-junction vector-current hypothesis (x = +-125 nm, each wall steps 79 nm); equal point
  dipoles too shallow (peak 1.14x broadside at |u| 0.81) -> fit 6 complex vector currents to the complex sphere field with held-out
  test; TM 1/q^2 can come from opposite-phase half-line envelope currents (not proof of a field jump); TM cancelling pitch for the
  grazing band 0.53-0.56 or 7-24 um, needs a chirped/weighted set.
- 13:25 DISPATCHED 266507 (runners/sweeps/farfield_sph_15um_junction_plain.py, 4 rows, default partitions, 1 concurrent, ~0.5 h each):
  plain TE shoulders softening the cavity-junction steps - right (x 180 nm, narrow wall) 10 nm, left (x -70, cavity wall) 10 nm,
  both, right 5 nm. Control = stored round-A plain far field. Pass: resolved >= 10 % reduction, lambda/W held.
- 13:50 SOURCE-LOCATION FITS (scratch junction_fit.py + general-basis variant: 24 vector currents per source, fitted to the complex
  sphere field, held-out = every 3rd phi column). Explained power plain TE / TM: centre only 0.51 / 0.47; junctions +-a (a = pitch/4)
  0.86 / 0.62; +-a,+-3a 0.97 / 0.77; +-a..+-5a 0.99 / 0.84; 21-point grid over +-10 um 0.75 / 0.91. => TE leak is born at the first
  three wall steps on each side of the cavity (six radiators, the junctions dominant; in an unbroken grating step radiations cancel
  coherently, at the defect the remainder survives). TM leak is distributed over +-10 um (envelope tail). Cancelling features on TE
  must act on those steps; extended antennas farther out miss (consistent with every tab-array result).
- 14:00 SAME FIT ON THE LAB DEVICE: Nt80x control: +-a 0.39, 5 steps/side 0.60, 21 pts +-5 um 0.74; +20 nm parent: +-a 0.15, 5 steps
  0.44, +-5 um grid 0.66. => the control leak = compact inner-step part + distributed part; the contrast knob removed most of the
  compact part (junction pair 0.39 -> 0.15); the residual is distributed over +-5 um (envelope-tail type, TM-like). Local added
  features cannot reach it; only a distributed change of the corrugation profile kappa(x) (its Fourier tail in the light cone at
  fixed mode width) can. Next math: analytic envelope design with radiation per stored energy as the cost.
- 14:20 GPT (s22): plain TE step-local basis = one shoulder per wall step (+-125, +-375, +-625 nm); no single feature is harmonic-selective
  by symmetry (all stay in the E l+m even / M l+m odd sector); M(1,0) alone + lambda/W needs >= 4 real knobs, the top-4 groups >= 10.
  QUEUED 266573 = junction_plain rows 4-8 (left half depth, shoulders at -5a,-3a,+3a,+5a, 10 nm), afterok 266507, 1 concurrent.
- LITERATURE (Opus agent): Johnson, Fan, Mekis, Joannopoulos APL 78 3388 (2001) = single-multipole cancellation by sweeping ONE knob
  next to the defect (root of a_m; Q x10-16); multi-multipole cancellation by a linear combination of perturbations NOT FOUND in any
  cavity work (ours would be new); 1-D records = Gaussian-envelope tapering (Quan & Loncar OE 19 18529), explained alternatively as
  mode matching at the terminations (Sauvan/Lalanne PRB 71 165118) - consistent with our leak born at the first wall steps.
- 14:35 ANALYTIC REACH MAP (Fable agent, scratch reach_map/, point dipoles in uniform oxide, gate: dipole at origin = pure l=1):
  a feature at x0 peaks at l ~ k x0 (l <= 5 > 98 % only for x0 <~ 0.5 um); no single feature > ~60 % in one harmonic. EXACT exclusion:
  x-even pairs reach only m-odd harmonics (class A), x-odd pairs only m-even (class B). Single-harmonic antennas from 2-4 step sites with
  signed weights: E(3,+-3) 86 % (2 sites, added-only), M(1,0) 83 % (3 sites alternating: tabs + notches), E(4,+-4) 76 %, M(3,0) 68 %;
  null-space 4-site pattern kills M(1,0) and E(2,+-2) together. Nt80x top harmonics E(4,+-4), M(3,0), M(1,0) are all m-even -> class B
  -> need x-odd sign patterns. Plan: the measured step-shoulder columns (266507 + 266573) replace the point-dipole model; synthesise
  the selective antennas as linear combinations and test the predicted single-harmonic one with one row.
- 15:00 FORWARD CITATIONS of Johnson et al. 2001 (Opus agent; Semantic Scholar 155, OpenAlex 134, Crossref 106, OpenCitations 105;
  classified from abstracts): ~15 use/extend the idea, mostly PhC-slab cavities tuned over 1-4 geometry knobs (Ryu/Notomi hexapole
  Q > 2e6; Takata 2023 H1 Q > 1e8 with 4 parameters). Karalis 2004 (OL 29 2309) cancels coupling to SEVERAL leaky channels at once by
  near-field orthogonality - closest to multi-channel. 1-D closest: Ding, Liu, Li 2012 (OE 20 28641) sums the radiation of every
  section interface coherently in a 1-D PhC reflector on a waveguide and designs minimum-radiation tapers = the same physics as our
  step-source fit; Watts 2002 (mode-matched Bragg resonator), Riboli 2005 (Si3N4 air-slit Bragg, first slit pairs optimised),
  Sauvan/Lalanne 2005. Gap: no paper solves perturbation amplitudes from a measured matrix of multipole coefficients (likely, not
  proven); in 1-D the interference/mode-matching idea is established, so a multipole framing must be set apart from Ding 2012.
- 15:20 HARMONIC RESPONSE MATRIX (Fable agent, scratch harm_matrix/, all stored Nt80x rows, solve_v4 --short normalisation): NO measured
  knob is harmonic-selective (max selectivity |Da_target|^2/|b|^2 = 0.10, contrast on E(3,+-3); bumps/subteeth/tabs <= 0.08, most <= 0.01).
  The contrast knob moves ALL top harmonics together at ~160-180 deg (a common-mode scaler of the l<=5 block, eta^2 0.32-0.44).
  Multi-knob solves on the control basis predict 0.42-0.44 but are noise fits (bump rows ~1.6x the floor, linearity 0.5-0.64);
  zeroing top-4/top-8 harmonics is not reachable within bounds. Tab rows raise P vs the +20 parent in every case. => selectivity has
  to be BUILT from signed multi-site patterns (reach map) on a measured step-site basis (266507/266573), not found among existing knobs.
- 15:40 FIRST JUNCTION ROW (266507_0, plain TE, MEASURED vs stored round-A control, Lorentzian norm): 10-nm shoulder on the narrow wall
  just right of the +125 nm step: P/Pc 0.958, Q_i 2.02e4 -> 2.11e4 (+4.5 %), dlam +0.040 nm, dW -0.049 um; |b|^2/|c|^2 0.0014,
  overlap eta^2 0.344 (Re<c,b>/|c|^2 -0.022) - the FIRST added feature that cancels, with contrast-like overlap. Linear optimum would
  need ~16x the depth (beyond the 79-nm step) -> deeper shoulders next if row 3 (5 nm) confirms linearity.
- 16:00 DISPATCHED 266742 (runners/sweeps/farfield_sph_15um_cavsize_plain.py, 4 cards, 1 concurrent): plain TE cavity width 820/780 nm and
  cavity length 270/230 nm (user: "check we're not just enlarging the cavity"). Tests: share of the shoulder response b_J_R inside the
  complex span{b_width, b_length} (near 1 => a cavity resize), and the best reduction from resizing alone. TRAP caught pre-dispatch:
  card field cavity_length_nm is IGNORED by to_device_kwargs (length comes only from cavity_neg_detuning_nm) - the length rows would
  have run at the default length; fixed + asserts that every size change reaches the builder kwargs.
- 16:20 CAVITY-CENTRE TAB on the Nt80x +20 parent (266486_0/1, MEASURED): 5 nm P/Pp 0.995, 2.5 nm 1.010; |b|^2/|c|^2 0.003 / 0.002
  (below the 0.012 per-row floor), Q_i unchanged, dlam +0.005 nm. NULL: a tab at the cavity centre does not touch the residual leak
  (consistent with the source fit: after contrast the residual is distributed over +-5 um, the junction pair holds only 0.15).
- 16:50 LEFT-JUNCTION SHOULDER (266507_1, 10 nm on the cavity wall right of the -125 nm step, MEASURED): P/Pc 1.025 (ADDS), overlap
  eta^2 0.18 with the adding sign, Q_i -2 %, dlam +0.010, dW +0.003. Right shoulder cancels (0.958), left shoulder adds -> the two
  junctions respond with OPPOSITE sign: not a symmetric cavity enlargement. The free 2-knob solve wants right tab +167 nm and left
  notch -164 nm (far outside linear range) = moving cavity-width material from the left end to the right end ~ an asymmetric
  cavity / phase-slip offset, which per-tooth shifts in inverse design can also express. Bounded (|alpha|<=1, lambda/W held): 0.973.
- 17:30 CAVITY WIDTH +20 nm (266742_0, plain TE, MEASURED): P/Pc 0.988 (|b|^2 0.0028, Re<c,b> -0.0072, dlam +0.045, dW -0.024) - one third of
  the right-shoulder gain with 2.5x the added material. Complex |cos| shoulder vs width response: J_R 0.58, J_L 0.70; share of J_R
  inside span(width) 0.34 (J_L 0.49). Length rows pending for the full span{width, length} test.
- 18:10 ADDITIVITY (266507_2, both junction shoulders 10 nm, MEASURED): P/Pc 0.9805 vs linear prediction c+b_R+b_L 0.9825; field residual
  ||b_LR-(b_R+b_L)||/||b_LR|| = 0.022; dlam +0.050 (sum +0.050), dW -0.046 (sum -0.046). Step-site features on plain TE ADD LINEARLY
  (unlike the Nt80x tabs, gate 0.34) -> the meeting's linear solve over step-site knobs is valid on the plain device.
- 18:40 CAVITY WIDTH -20 nm (266742_1, MEASURED): P/Pc 0.990 (+20: 0.988); Re<c,b>/|c|^2 -0.0075 / -0.0073 (same sign both ways ->
  no first-order width lever; the width sits near a flat point); pair residual 0.24. Width resizing gives ~1 %, the right-junction
  shoulder 4.2 % with less material -> the shoulder is not a width resize. Length rows pending.
- 19:30 NOTCHES ON THE LAB DEVICE (266486_4/5, MEASURED vs the +20 parent, cavity-trace norm): 7 notches cut into the left-arm wall at the
  half-period tab sites (x -330 .. -4057 nm), 5 nm: P/Pp 0.896, Q_i 7.49e5 -> 8.18e5 (+9 %), dlam -0.065 nm, dW -0.002 um,
  |b|^2/|c|^2 0.084; 2.5 nm: 0.934, Q_i +6 %, dlam -0.027. Pre-registered first-order hypothesis 0.92 / 0.95 -> measured better.
  vs the original control: 0.684 x 0.896 = 0.613 (DERIVED), Q_i +53 %. Sub-period notches = OUTSIDE the inverse-design parameter space.
  Depth not linear (|b|^2 ratio 5.8 for 2x). GPT gate (>= 10 % resolved, lambda/W held) met at dy 50; confirmation pending.
  Plain right shoulder 5 nm (266507_3): 0.969 vs 10 nm 0.958, linearity residual 0.64 -> sub-linear in depth (plain responses ~1e-3).
- 19:35 DISPATCHED 266892 (cavity rows 6-7: notch 7.5 / 10 nm, A100, 2 concurrent) and 266894 (cavity_dy37 row 4: the 5-nm notch at
  y_mesh_cells 20, vs the stored dy-37 +20 parent 265968).
- 20:00 USER: "check we're not just making a more optimal (bigger) cavity; find the optimal cavity first and add features on top".
  Memory: the rect-1050 optimum (loss -30 %, flat 1040-1075, + see-saw -31 %) was the plain TM device (pitch 516.83 / corr 400 / W800),
  NOT TE; plain TE cavity width was never scanned beyond today's +-20 nm (~1 %). DISPATCHED 266913 = cavsize_plain cards 4-9: plain TE
  cavity width 850/900/950/1000/1050/1100 nm (2 concurrent). Next: shoulders/notches on the optimal-width TE cavity.
- 20:15 CAVITY LENGTH +20 nm (266742_2, MEASURED): P/Pc 1.039 (worse), dlam +1.10 nm, dW -0.283 um, Q_i 1.94e4 (-4 %). 266742_3 (length
  -20) died at 21 s with a bare 'in run:' (no solve; layout saved) = license/startup race signature -> resubmitted once as 266940.
  Span test: 56 % of the right-shoulder response lies in the complex span(width, length) (upper bound: the length coefficient comes out
  imaginary, which no real resize can produce). Lambda-equivalence argument: the shoulder's +0.040 nm shift equals ~+18 nm of cavity
  width (worth ~1 % radiation) or ~+0.7 nm of length (worth ~+0.1 %, and lengthening RAISES radiation) -> an effective resize can
  account for at most ~1 of the shoulder's 4.2 %.
- 20:40 266742_3 / 266940_3 (cavity length -20) both died at 17-21 s on node n317: layout _p0.log = FlexNet -4,132 "only 10 of the
  requested 12 tasks" - n317's GPU has 188 SMs -> 12 seats per solve. Resubmitted pinned to A100 (7 seats) as 266947. Lesson added to
  CLAUDE.md 6 and the athena-preflight skill.
- 20:45 CAVITY WIDTH 850 nm (266913_4, MEASURED): P/Pc 0.948, Q_i 2.14e4 (+6 %), dlam +0.110, dW -0.074 -> the TE cavity optimum lies at larger width (like TM 1050); the ladder decides; the shoulder must then be tested ON the optimal cavity.
- 21:20 CAVITY WIDTH 950 nm (266913_6, MEASURED): P/Pc 0.918, Q_i 2.22e4 (+10 %), dlam +0.270, dW -0.212 um. Trend continues (800 1.000, 850 0.948, 950 0.918); 900/1000-1100 pending.
- CAVITY WIDTH 1000 nm (266913_7, MEASURED): P/Pc 0.912, Q_i 2.24e4 (+11 %), dlam +0.335, dW -0.273 um. Ladder so far 800 1.000 / 850 0.948 / 950 0.918 / 1000 0.912 (flattening); 900, 1050, 1100 pending.
- CAVITY WIDTH 900 nm (266913_5, MEASURED): P/Pc 0.953, Q_i 2.14e4, dlam +0.190, dW -0.134. Ladder 800 1.000 / 850 0.948 / 900 0.953 / 950 0.918 / 1000 0.912 - NOT monotonic (900 above 850): oscillation or noise of a few points; treat single-point differences < ~0.01 as unresolved.
- NOTCH DEPTH SCAN (266892, MEASURED vs +20 parent): 2.5 nm 0.934, 5 nm 0.896, 7.5 nm 0.917, 10 nm 0.939 -> optimum near 5 nm (Q_i 8.18e5). CAVITY WIDTH 1050 nm (266913_8): 0.899, Q_i 2.28e4 (+13 %), dlam +0.395, dW -0.339; 1100 pending.
- DISPATCHED 266993 (farfield_sph_15um_onopt_plain, 3 rows, A100): 1050-nm cavity + right shoulder 10 nm; 1050 + left-arm notch set 5 nm; 800 + same notches (owner: do features still help on the optimal cavity?).
