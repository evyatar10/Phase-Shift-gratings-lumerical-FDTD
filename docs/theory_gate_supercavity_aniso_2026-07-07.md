# Theory gates: single-cavity/supercavity FW + anisotropic cladding (2026-07-07)

Zero-GPU gates run at the user's request ("do some theory first before spending
GPU") on the two remaining non-active novel ideas. Both use the MEASURED stack
mode field (`tm_field_export/.../_EZSLICE.mat`). Scripts:
`python_tools/phase0_supercavity_fw.py`, `python_tools/phase0_aniso_cladding.py`.
Active gain / time-modulation are OFF the table (user: "problematic").

## 1. Single-cavity / supercavity Friedrich-Wintgen — MIRAGE (do not run)

Idea: the two-cavity FW failed (job 118734) because the two cavities' radiation
lobes didn't overlap (rho too low). Could ONE cavity with two co-located modes
(Rybin/Bogdanov "supercavity" quasi-BIC) reach the dark state instead?

Gate result — the blocker is NOT overlap this time:
- Odd (1-node) partner: rho = 0.05 — parity-orthogonal, no interference (expected).
- Even (2-node) partner: **rho = 0.88 — CLEARS the rho>=0.82 CMT gate.** Formal
  dark-state floor (1-rho^2)*loss0 = 0.012. So a mode and its own even higher-order
  partner DO radiate into the same channel enough to interfere.

It still fails for two structural reasons that overlap can't fix:
1. **No partner exists.** A single pi-shift defect has one gap state; higher
   longitudinal states are expelled into the bands. Making a second gap mode needs
   a longer/coupled-defect cavity whose two modes are split by the cavity FSR (not
   degenerate). Tuning them toward degeneracy gives two spectral poles = **two
   peaks (violates single-resonance)** and/or a **wider mode (violates ~1% budget)**.
2. **A transmission port cannot harvest a BIC.** At the exact FW-BIC point the dark
   mode is decoupled from radiation AND from the waveguide port — invisible in
   T(lambda). Back off to a quasi-BIC to restore port coupling and you reintroduce
   loss and pull the bright partner into the window. This IS the "resonance drains
   instead" failure the two-cavity FDTD showed. The supercavity trick works for a
   Mie SCATTERER read in reflection (Mie + Fabry-Perot modes, co-located, tunable
   by aspect ratio); our device is a single-mode TRANSMISSION cavity — structurally
   incompatible.

Verdict: consistent with and explains the two-cavity FDTD failure. **Not worth GPU.**

## 2. Anisotropic / low-index cladding — WORKS, but is the trench's mechanism

Idea: surround the mode with a cladding of lower EFFECTIVE lateral index (an SWG
form-birefringent metamaterial, or just lower-index material) to shrink the
lateral light-line kc' = n_eff*k0 and cut off the near-grazing leak by TIR.

Gate result (ceiling from the measured lateral radiation spectrum):
- The lateral leak is edge-piled: ~32-46% of it sits at |kx|/kc >= 0.90 (grazing).
- n_eff 1.35-1.30 (routine SWG birefringence) cuts ~39% of the lateral flux ->
  dLoss ceiling ~ +0.012 (0.0545 -> ~0.042).
- n_eff -> 1.0 (full air) ceiling ~ +0.015.

This **matches the air trench's measured -0.012** — because it is the SAME lever
(light-cone shrink / TIR), just applied as a surrounding layer instead of two
discrete walls. Advantages over the trench: can be stronger (surrounds the mode)
and is standard planar SOI fab (SWG cladding). **But it does NOT answer "novel
physics beyond cladding engineering"** — it IS the refined form of it.

## Bottom line (now theory-backed, not just asserted)

For a single-resonance, fixed-width, planar, uniform-height, passive (no gain /
no time-modulation) device, the loss levers are exactly two, and neither is new
physics:
- **Cladding / light-cone engineering** (air trench, or anisotropic SWG cladding):
  real, ceiling ~ -0.012 to -0.015. The user is right that this is "cladding
  engineering," not a novel interference mechanism.
- **Envelope optimization** (counterdiabatic / inverse design): a loss-vs-width
  PARETO, reachable by inverse design, not unique.

The genuinely-different interference escapes (two-cavity FW, single-cavity
supercavity FW, external secondary scatterer) are now all closed — two by FDTD,
the supercavity by theory (no co-located degenerate gap partner + BIC/port
incompatibility). Remaining true-novelty routes all require relaxing a stated
constraint (width Pareto, height/back-reflector, or platform change).
