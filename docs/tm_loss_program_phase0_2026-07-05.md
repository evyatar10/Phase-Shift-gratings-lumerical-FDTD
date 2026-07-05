# TM loss program — Phase 0 (theory + pending results), 2026-07-05

Tool: `python_tools/lateral_radiation_theory.py` (zero GPU, zero license).
Success bar (user, 2026-07-05): headline ΔT ≈ +0.05 vs champion (T 0.9165 → ~0.966,
i.e. recover ~60% of the remaining 8.2% loss); smaller confirmed gains still reportable.

## 0. anti_moment_cavity job 117814 (accurate mesh) — fetched

- Family B width ladder: shallow parabola, flat optimum 1050–1075 (champion confirmed);
  in-study jitter (1052 partner) ≈ 2e-4.
- Family A zero-area tooth pair (wide tooth ±1 = 1000+δ, ±2 = 1000−δ on the 1050 base):
  real signed trend. +δ helps, saturating: loss 0.0823 → **0.0810 at δ=+20/+30**
  (−1.6% rel, ~6× floor, fwhm_m +0.14%). −δ hurts progressively (+0.0048 at δ=−30).
- **New in-scope best: W1050 + ptw (1020, 980): loss 0.0810, T_res 0.9179.**
- Registered prediction said "linear-in-δ if residual local moment exists" — it did;
  now harvested and saturated. Remaining local budget after this: ~nothing above floor.
- Figure: `results_from_athena\anti_moment_cavity\anti_moment_cavity_summary.png` (+.fig)

## A. EIM n_eff and the "lateral leakage" question

Vertical TM slab (350 nm): n_eff 1.5911. Lateral EIM: n_eff(600)=1.4989,
n_eff(800)=1.5174, n_eff(1000)=1.5315, n_eff(1050)=1.5345, n_eff(1400)=1.5505.
Mean(600,1000)=1.515 vs measured Bloch 1.5078 (EIM overestimates ~0.5%, fine).

**Verdict:** even the narrow tooth is comfortably guided (margin +0.055). Our fully
etched ridge in uniform oxide has NO slab layer → no guided TE-slab continuum → the
literature "magic-width / lateral-leakage-BIC" mechanism does NOT transfer literally.
The loss is light-cone radiation into the 3D oxide continuum. What survives from that
literature family: (i) directional two-edge interference (B), (ii) the polarization
question (does the sidewall corrugation polarization-convert the radiated field —
Phase 1 measures this).

## B. Two-edge interference vs the measured cavity ladder

The measured ladder (max ~W800 → min 1050–1075 → worse by 1400) has period ~1.0–1.1 µm.
Two-sidewall interference needs ky = 2π/period ≈ 5.7–6.3 rad/µm ≈ the light-cone edge
kc = 5.82 rad/µm ⇒ **only BROADSIDE in-plane radiation (θ≈90° from the axis) is
(marginally) consistent**; near-axial 10° radiation would need a ~6 µm period and is
excluded as the ladder driver.

**Registered discriminator (cheap, 1 FDTD row):** two-edge interference predicts a
SECOND loss minimum near W_cav ≈ 2100–2200 nm; a pure local moment-null predicts
monotonic worsening past 1400. → row added to the 2c study.

## C. Near-cavity side reflectors (user request) — feasibility numbers

- Quarter-wave SiN strip in oxide = 198 nm wide; normal incidence (broadside channel):
  R = 0.09 / 0.31 / 0.53 / 0.72 / 0.91 for 1/2/3/4/6 strips (quarter-wave gaps 270 nm).
  **Gate passed: ≥0.2 with 2 strips.**
- Broadside recycling phase period in strip offset d: π/kc = **0.54 µm** → a d-scan
  samples the full phase inside the converged y=6.8 µm box. Near-axial (10°) period is
  3.11 µm → cannot d-scan in-box; phase must come from strip width/count there.
- Grazing (near-axial arm channel, θ_i≈80°): single-interface R_s=0.475, R_p=0.240 —
  high, but geometry is hard (retro-reflection of a near-axial lobe needs end-on or
  retro-diffractive structures, not a parallel strip).
- Drain bound: guided-mode lateral tail amplitude e^(−γd), γ=1.75 µm⁻¹ → 0.17/0.07/0.03
  at d=1.0/1.5/2.0 µm. Keep strips at d ≥ 1.2 µm; include a drain-discriminator row.

**Honest ceiling logic:** the cavity-local radiating share is ~30% of 0.082 loss
≈ 0.025 T; a near-cavity reflector recycling HALF of it ≈ +0.012 T. Good, but the
+0.05 T headline needs the ARM share too → the full-arm strip variant (length ~84 µm)
is the one that can in principle reach the bar, IF Phase 1 confirms in-plane dominance
and the phase stays coherent along the arm.

## D. SSH gap dimerization — PAPER-KILL (do not dispatch)

1D TMM calibrated to the device (pattern-REPLACEMENT π-shift topology — the cavity
replaces the innermost narrow slot; resonance 1558.4 nm ✓, envelope fwhm 15.41 µm ✓).
Validation on a known FDTD outcome: distributed π-shift → light-cone weight +348%
(FDTD: +21–39% loss) — correct sign, exaggerated magnitude (crude proxy, ratios only).

Dimerization ±δ/2 on K innermost gaps/side (δ=10/20/40 nm, K=4/8/16):
**every row INCREASES the in-cone weight** (+4% … +374%); larger δ also violates the
fwhm bound (±5–8%). Physics: alternating gap detuning = envelope kinks at the defect —
the same failure mode as the distributed shift, not the SSH band-repositioning hoped
for (that would require redimerizing the WHOLE arm, which strengthens the mirrors and
narrows the mode → out of scope anyway).

Model caveats (honest): 1D scalar proxy; no transverse physics (cannot see the cavity
ladder's transverse moment-null); model peak T=0.68 (mirror asymmetry of the replaced-slot
topology is exaggerated in 1D — real device T=0.92); use SIGN/ratio only. The kill
stands on the validated sign + the constraint violation.

## Bug journal (for reuse of the TMM)

- π-shift topology: the device is a half-period pattern REPLACEMENT (merged
  wide|cavity block), NOT a quarter-wave insertion. A λ/4 insertion between periodic
  arms has NO in-gap state in 1D TMM (its FP phase lands exactly between the even and
  odd conditions, ±π/2) — building it wrong hides the resonance entirely.
- The defect peak is pm-wide at N=80: brute-force λ scans cannot find it; use the FP
  round-trip phase condition (2φ_cav + arg r_L + arg r_R = 2πm) and bisect.

## Consequences for the program

1. **2e (SSH dimerization): killed on paper.** Zero GPU spent.
2. **2c (cavity L×W)** gains two theory-motivated rows: W_cav ≈ 2100 (two-edge second
   minimum discriminator) and the Family-A direction is already harvested (0.0810).
3. **2b (reflector)**: broadside variant is feasible on paper (2 strips → R≈0.3,
   phase scannable in-box); prioritized near-cavity first (user request), full-arm
   extent is the only route to the +0.05 T headline.
4. **Phase 1 polarimetry** decides: in-plane vs vertical split, broadside vs
   near-axial angle, and polarization of the radiated field.
