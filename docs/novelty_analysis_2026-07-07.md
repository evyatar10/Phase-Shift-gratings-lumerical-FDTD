# What is genuinely novel (beyond tooth-shift + apodization)? — theory + research

Written 2026-07-07 (autonomous) answering the user's questions: (1) is there a
loss-reduction route that is NOT reachable by inverse design of tooth
shift/width + apodization? (2) Green's-function theory of the external
scatterer's +0.0026. (3) Is left-right symmetry actually optimal? Research +
first-principles, then experiments to test the survivors (not just assert).

## 0. The controlling physics (why most planar tricks are envelope optimization)

The defect mode is a localized resonance with carrier at kx = ±β (β = π/Λ =
1.508·k0), which is OUTSIDE the light cone (kc = n_clad·k0 = 1.444·k0 < β). So
the periodic grating itself does NOT radiate. The **radiative loss is entirely
the mode ENVELOPE's Fourier tail reaching the light-cone edge** at kx = kc:
  loss(in-plane) ∝ |Ê_envelope(kc)|²,   kc a distance β−kc ≈ 0.25 rad/µm below
  the carrier; envelope Fourier width ≈ 1/fwhm ≈ 0.067 rad/µm ⇒ we live ~3.7
  envelope-widths out on the tail.
Anything that keeps ONE localized resonance at fixed width can only reshape that
envelope ⇒ **envelope optimization ⇒ exactly the inverse-design space** (tooth
shift/width, apodization, the counterdiabatic profile). This is why the shape
studies (barrel/hann/tri/ellipse/wedge/tilt) were neutral-to-worse: changing the
unit-cell tooth SHAPE does not touch the envelope tail (the carrier is out of
cone). The counterdiabatic profile is the best point found IN this space — and
the user is right that inverse design would reach it.

**There are exactly three ways OUT of envelope optimization** (from the program
brief §2, now tested):
- (A) spectral-null envelope shaping — the derived "golden profile"; a local
  optimum, spent.
- (B) interference with a SECOND resonance / an external secondary source (BIC,
  scatterers) — *evades* the envelope wall. **TESTED THIS SESSION → FAILED (§2).**
- (C) relax a constraint (width or height/material) — off the table by the
  user's "same device" rule, EXCEPT as the known width-cost Pareto.

## 1. Green's-function theory of the external scatterer (why +0.0026, can it grow?)

A scatterer at r_s is a secondary source: it re-radiates α(ω)·E_mode(r_s), and
the total radiated field is A(kx) = A0(kx) + α·G(r_s→far)·E_mode(r_s). The
best-case cancellation (optimize the COMPLEX strength α) leaves residual
  1 − |⟨A0, S⟩|² / (|A0|²·|S|²),  S = G·E_mode(r_s)
= the placement-map ceiling computed in `phase0_greens_overlap.py`: **one pair
cancels ≤ 33% of the in-window in-plane leak (≤ +0.008 T), two pairs ≤ +0.016.**

**Why the real passive cylinder only gets +0.0026 (and mostly ADDS loss):** a
low-contrast dielectric cylinder has α ≈ REAL and small (far below its own Mie
resonance; SiN/oxide Mie Q ~ 1–3). The optimal α is COMPLEX with a specific
phase. A fixed-phase α equals the ceiling only where S happens to be in phase
with −A0; elsewhere it adds |αS|² (parasitic drain) — which is why batch-1's
scatterers, placed at the *optimal-α* map sites, RAISED loss (wrong phase there).
The +0.0026 anchor was a lucky site where the fixed real α partially aligned.

Two ways to supply the missing α PHASE, both tested and both failing:
- a RESONANT element (α sweeps 0→π across its resonance) = the FW-BIC side
  cavity → **FAILED (§2): the resonance drains instead.**
- a CLUSTER of passive scatterers whose POSITIONS synthesize the needed phase
  from fixed real α's. **Not yet tested → phase-2 tests it** (design in
  `phase0_greens_cluster.py`), with the honest caveat that the in-plane overlap
  model is optimistic (it ignores vertical scatter + guided-mode back-reflection,
  the very parasitics that sank the single scatterers).

## 2. FW-BIC / interference route — TESTED, FAILED (job 118734)

Two side-coupled π-shift cavities, detuning × coupling scan. Every coupled cell
raised device-1 loss dramatically (loss 0.35–0.47, peak-T 0.4–0.6 vs the 0.112
weak-coupling reference), monotone worse with stronger/nearer coupling, NO
subradiant dip anywhere. This is the low radiation-pattern-overlap (ρ) failure
the CMT flagged: the two cavities' radiation lobes don't match, so coupling opens
a big loss channel instead of a dark state. Combined with the scatterer failure,
**both "second radiator" routes (B) are empirically dead for this device.**

## 3. Is left-right (mirror) symmetry optimal? — YES, provably (for fixed width)

The symmetric cavity mode is EVEN about the defect ⇒ its radiated amplitude
A0(kx) is EVEN in kx. A structural perturbation δε adds δA ∝ FT[δε·E_mode]:
- SYMMETRIC δε (even): δε·E even ⇒ δA even ⇒ can cancel A0 (this is the whole
  tooth-shift/apod/CD space).
- ANTISYMMETRIC δε (left arm +, right arm −): δε·E odd ⇒ δA ODD ⇒ orthogonal to
  the even A0. The cross term ∫_{|kx|<kc} A0*·δA dkx vanishes by parity, so
  ∫|A0+δA|² = ∫|A0|² + ∫|δA|² ≥ ∫|A0|².
⇒ **any small left-right-asymmetric perturbation can ONLY INCREASE radiation.**
This is exactly why the earlier antisymmetric-depth ("anti-moment") study was
neutral-to-worse — now explained, not just observed. Mirror symmetry is optimal
in the small-perturbation (fixed-mode) regime. Asymmetry helps only if allowed to
RESHAPE the mode (large asymmetry) — which changes the width / splits the
resonance, violating the constraints. **Phase-2 confirms this empirically** on the
current-best device with a short antisymmetric-perturbation scan (existing
`asym_inner_dw_delta` knob) — a foregone conclusion by the argument, run only as
a check. (Literature "asymmetric cavity → higher Q" is the mode-reshaping regime,
and/or up-down (vertical) asymmetry for unidirectional guided resonances, which
redistributes radiation between up/down rather than reducing it — not our lever.)

## 4. Research scan — what the field does (2023–2026) and why it doesn't port here

- **BIC / merging-BIC / quasi-BIC in gratings & metasurfaces** (many 2024–25
  papers): ultrahigh Q by merging symmetry-protected + accidental BICs, or by
  tuning unit-cell Fourier harmonics. BUT these are PERIODIC leaky modes
  radiating at the Γ point (kx=0); our loss is a localized-defect ENVELOPE tail
  near the band edge — a different radiation mechanism. The BIC recipes don't
  transfer to a single defect cavity. (Our π-shift IS already the SSH topological
  mid-gap defect; topology protects its existence/frequency, not its radiative Q.)
- **Unidirectional guided resonance (UGR) via broken up-down + C2 symmetry**:
  makes radiation one-sided (up vs down), not smaller — and needs an asymmetric
  vertical stack (non-planar). Not a loss-reduction lever here.
- **Non-Hermitian / exceptional points**: enhance sensitivity, not passive Q; and
  for our metric loss = 1−T−R, adding material absorption only makes 1−T−R worse.
  Not applicable.
- **Machine-learning / GA optimized nanobeam cavities** (Q ~ 10⁶–10⁸): this IS
  inverse design of the envelope — confirms the user's point that our CD win sits
  in the inverse-design-reachable class.

## 5. Honest bottom line + what phase-2 still tries

The physics + this session's failures point to a strong conclusion: **for a
single-resonance defect cavity at fixed mode width, planar and uniform-height,
there is very likely NO loss-reduction lever beyond envelope optimization** (the
CD/inverse-design class). The genuine escapes (second-radiator interference,
external secondary source) FAILED in FDTD; symmetry breaking is provably
counterproductive; BIC/EP/UGR recipes don't port to a localized defect mode. The
only remaining true-novelty escapes require relaxing a constraint: **width**
(the known Pareto: +3% width buys more loss reduction) or **height/contrast**
(a back-reflector layer, higher-index core, or a photonic-bandgap cladding —
none of which are "the same device").

That said — to make this EMPIRICAL, not just an argument, phase-2 launches a
multi-angle array that actually tries the last untested planar ideas on the
confirmed stack baseline:
1. **Antisymmetric-perturbation check** (confirm §3 on the best device).
2. **Green's-function passive scatterer CLUSTER** (§1 — positions solved for the
   real α, the user's requested direction; honest low-confidence).
3. **Leaky-wave reflector**: a periodic scatterer array in the cladding beside the
   arms, spaced to Bragg-reflect the near-axial leaked wave back toward the mode
   (modify the continuum / recapture — the one "external structure" variant not
   yet tried).
4. **Co-located strong 2nd-harmonic**: an extended every-other-tooth modulation to
   try to open a genuine second band and get single-device FW interference (the
   partner-type-2 the brief named; batch-1 only did tiny amplitudes).
If all four are null, the bottom line above stands as an empirical result, and
the recommendation is: accept the CD/inverse-design envelope optimum, and pursue
loss reduction only via the width-cost Pareto or a platform change.
