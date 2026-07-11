# Leaky-field energy pattern & innermost-tooth cancellation ceiling (TM) — 2026-07-08

Zero-GPU theory gate for the "exotic symmetric innermost-tooth to recycle/refract the
leak" idea. Companion typeset PDF + figure:
- `docs/theory_innermost_recycling_2026-07-08.pdf`
- `docs/theory_innermost_recycling_2026-07-08.png`
- generator: `python_tools/theory_innermost_recycling.py` (reads on-disk data, no run)

## The radiated-field energy pattern (derived, then fit)

Mode = Bragg carrier under a cusped envelope: `E(x)=A(x)u(x)cos(beta x)`, `A(x)=e^{-kappa|x|}`,
`beta=n_eff k0`. Light escapes only for `|kx| < kc = n_clad k0`. Radiated amplitude is the
mode Fourier component inside the cone, `E_rad(kx)=½[Ã(kx-beta)+Ã(kx+beta)]`, `Ã(k)=2kappa/(kappa²+k²)`.

**Energy pattern:** `P_rad(kx) ∝ |Ã(kx-beta)|² = (2kappa)²/[kappa²+(beta-kx)²]²`, `0<kx<kc`
— a Lorentzian-squared rising to the grazing edge `kx→kc`. Total loss `∝ ∫_0^{kc} P_rad dkx`,
set by the edge value `Ã(Δk)`, `Δk=beta-kc`.

## Fit to on-disk data (N80 TM, W800 baseline + stack)

| quantity | value |
|---|---|
| lambda_res | ~1558 nm |
| n_eff / n_clad | 1.5078 / 1.4440 |
| beta / kc | 6.0794 / 5.8221 µm⁻¹ (beta/kc = 1.044, carrier OUTSIDE cone) |
| Δk = beta-kc | 0.2573 µm⁻¹ = 0.064·k0 |
| kappa (from mode FWHM) | 0.0446 µm⁻¹ (Δk/kappa ≈ 5.8) |
| measured spectrum peak | u_x = 0.990 (grazing) — matches theory |
| model fit to in-cone spectrum | R² = 0.80 |

The envelope is NOT a clean single exponential (per-tooth "beads" ride under `e^{-2kappa|x|}`)
— direct evidence the radiation is arm-distributed, not defect-local.

## Cancellation / recycling condition and the ceiling

Symmetric innermost pair at ±x1 = secondary source `E_s(kx)=2 s A(x1) cos(kx x1) g(kx)`.
Destructive-outside condition `E_rad+E_s=0` across the lobe; best single pair from
`min_s ∫_0^{kc}|Ã(kx-beta)+2 s A(x1) cos(kx x1)|² dkx`. Required phase `arg(s) ≈ 180°`
(π out of phase with the leak at x1).

**Ceiling (two independent estimates, agree):**
- Spatial origin: only **15%** of the in-cone leak originates within ±1 tooth (±3: 29%, ±8: 43%).
- kx-projection: a symmetric innermost pair cancels **~1.3%** of radiated power; 3 pairs ~15%
  (OPTIMISTIC — ignores the scatterer's own added radiation).

**Verdict:** innermost-teeth-ONLY ceiling ≈ **ΔT +0.001 to +0.006**, consistent with the prior
scatterer plateau (+0.003). Marginal but nonzero. The dominant near-grazing leak (~70% in the
arms) needs ~1 µm features placed IN THE ARMS to phase-match — outside the innermost-only scope.

## Candidate shape (theory-motivated, for the agreed single run)

Not an apodization/taper. A **localized double-lattice element**: give the innermost tooth pair a
secondary sub-edge offset by ~quarter Bragg period so its vertical scattering is ~π-shifted
(K&H / Noda destructive-interference mechanism), realized as a symmetric custom polygon via
`add_shaped_tooth` (extend to accept `inner_tooth_vertices`). Scan the offset to sweep `arg(s)`
through π and locate the (small) cancellation optimum. Compare to a rect-innermost control at
identical numerics; survivors confirmed at `accurate` mesh.
