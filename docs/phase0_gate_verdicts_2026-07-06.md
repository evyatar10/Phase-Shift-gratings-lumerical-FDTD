# Phase-0 Gate Verdicts — BIC / Kerker / Counterdiabatic (2026-07-06)

Zero-GPU theory pass for the program in `loss_program_bic_scatterer_2026-07-06.md`.
All computed locally from existing data (accurate-mesh stack field export
`tm_field_export` job 118462, polarimetry job 117907, measured stack/pair dose
curves). **Nothing was built or dispatched.** Scripts (rerunnable):

| # | route | script | verdict |
|---|-------|--------|---------|
| 4.1a | symmetry-protected BIC | `python_tools/phase0_bic_cmt.py` (Part A) | **FAIL — drop** |
| 4.1b | Friedrich–Wintgen single-location BIC | `python_tools/phase0_bic_cmt.py` (Part B) | **PASS (conditional on ρ ≥ 0.82)** |
| 4.1c | vertical anti-phase out-coupler (new, from 4.4 framing) | `python_tools/phase0_bic_cmt.py` (Part C) | **PASS (order-of-magnitude)** |
| 4.2 | Kerker/Huygens scatterers | `python_tools/phase0_kerker_mie.py` | **SPLIT: backward-Kerker IMPOSSIBLE; forward-Huygens PASS** |
| 4.3 | counterdiabatic apodization | `python_tools/phase0_counterdiabatic.py` | **MARGINAL — letter-pass, spirit-fail** |
| 4.4 | Green's-function overlap (aiming input) | `python_tools/phase0_greens_overlap.py` | done — feeds all above |

---

## 4.4 Green's-function overlap (inputs for everything else)

From the stack's accurate-mesh 2D field (±12 µm × ±3 µm, z=0):

- **y-parity of the in-cone radiating amplitude: 100.0% EVEN / 0.0% odd** (stack
  and W800 alike). x-parity: stack 49% even / 51% odd (arms are not x-mirror
  images); W800 82/18.
- **Single scatterer-pair cancellation ceiling** (optimal complex strength,
  point source, anywhere in the window): **33% of the in-plane in-window leak
  → max ΔT ≈ +0.008** on the stack. Two pairs: 67% → **max ΔT ≈ +0.016**.
  Best site x₀ ≈ +0.25 µm, y₀ ≈ 0.65 µm (hugging the cavity); the legacy
  x=0.81 µm site reaches 24% (vs 33% best) — it was a good pick.
- Fringe spacing of good sites ~0.8 µm (mid-angle content), consistent with the
  window-resolvable part of the spectrum.
- **Caveat (registered):** the ±12 µm export window gives Δk = 0.26 rad/µm; the
  guided peak sits ~1 bin outside the cone edge → the near-axial bins
  (|ux| > 0.9), where polarimetry says the stack's residual concentrates, are
  under-resolved. The ceiling numbers cover the resolvable (broadside-to-mid)
  part; a cheap 1-row ±40 µm 1D line export would sharpen them if wanted.

## 4.1a Symmetry-protected BIC — FAIL, drop

The only mirror under which the continuum coupling has definite parity is
σ_y (measured 100% even). A y-odd operating mode is indeed exactly decoupled
from that channel — **and from the y-even input port by the same integral**:
protection and transmission are killed together. σ_x: continuum carries both
parities (49/51) → nothing to forbid. σ_z: up/down continuum has both
parities. Γ-point BIC machinery doesn't apply at our band-edge operating
point (the periodic part already doesn't radiate; all loss is envelope
broadening). No realizable single-resonance protected variant exists.

## 4.1b Friedrich–Wintgen at one location — PASS (conditional)

2×2 non-Hermitian CMT with measured rates: g_tot = 1.109 nm, radiative
g₁ = 0.031 nm (Q_rad ≈ 50k; extraction self-consistent: x = 0.0279 reproduces
both T = 0.9449 and loss 0.0543≈0.0545 with R ≈ 0).

**The lossy-partner regime does exactly what the brief hoped.** With a
deliberately LEAKY co-located partner (g₂ = 2–15 nm ≫ g₁), the dark state is
automatically mode-1-dominated (admixture g₁/g₂ ≲ 1% → width guard free) and
the FW detuning is automatically LARGE (D ≈ κ·√(g₂/g₁) ≈ 15–40 nm → partner
out of window, and broad/shallow besides → **single usable resonance**).
Representative point (g₂ = 5 nm, ρ = 0.9): κ = 1.1 nm → 5.3× radiative
suppression, partner 16 nm away, admixture 0.5% → **predicted loss
0.0545 → 0.011, T → 0.989**. Feasible κ range 0.5–4 nm = evanescent
side-coupling territory.

- **Partner shortlist:** (1) short side-coupled parallel strip Bragg cavity
  (few periods, gap 0.3–0.8 µm — the side-by-side builder already makes this);
  (2) second-order-corrugated patch near the defect. NOT a cladding Mie rod
  (SiN/oxide rod Q ~ 1–3, no usable resonance).
- **The registered risk:** channel-pattern overlap ρ. Max suppression =
  1/(1−ρ²); **3× requires ρ ≥ 0.82**, unknowable in CMT — the FDTD scan IS the
  ρ measurement. Below ρ = 0.8 the route caps at ~2.8×; at ρ = 0.6, 1.6×.
- Polarization-agnostic: same map applies to TE (higher in-plane ceiling).

## 4.1c Vertical anti-phase out-coupler (new; falls out of the 4.4 framing)

Because the resonance sits at the band edge (β ≈ π/Λ), a weak every-other-tooth
width alternation (2Λ superperiod) is a first-order VERTICAL out-coupler acting
on the resonant envelope. Matching the existing vertical leak
(0.38 × 0.0545 = 0.021) needs only amplitude ratio ~0.14 → nm-scale
alternation. Two knobs (amplitude, which-tooth phase) against one complex
target → a small 2D scan can null the vertical channel IF the leak is
spatially coherent (its near-axial spikes suggest yes). **Only planar handle
on the 38% vertical share besides the FW-BIC itself; same builder knob as
existing width lists.** Risks: in-plane back-action (resonance re-trim),
unmeasured leak coherence.

## 4.2 Kerker/Huygens — backward impossible, forward strong

Validated 2D cylinder Mie (Rayleigh limits + unitarity checks pass), SiN in
oxide, m = 1.364:

- **Backward (anti-Kerker) directionality DOES NOT EXIST at this contrast:
  best B:F = 0.95:1** (never backward-dominant, any radius, either
  polarization; with usable strength only 0.6:1). The "throw it backward"
  version of the brief is closed by math — low-contrast dielectrics are
  forward-biased, period.
- **Forward-Huygens is strong and cheap.** Device-TM (E∥axis): F:B = 2.3:1 at
  the already-tested r=150, **7.8:1 at r=200, local optimum ≈65:1 at r=250 nm**
  (Q_sca 1.7). Device-TE: **sharp Huygens point at r = 260 nm, F:B ≈ 10⁵:1**
  (Q_sca 1.0). 3D-sphere cross-check agrees (first Kerker at r≈345 nm sphere).
- Consequence for design: the plain-cylinder premise ("omnidirectional
  monopole") was only true for r ≲ 100 nm; the r=200 row already had
  directionality. The new content is placing near-optimal forward-Huygens
  scatterers (TM r≈250, TE r≈260) on the 4.4 map's best sites with the phase
  arcs. **Ceiling unchanged: the 4.4 placement bound (+0.008 one pair, +0.016
  two pairs) is what directionality can approach, not beat.** Finite-height
  truncation → treat radii as bracket centers (±30% scan).

## 4.3 Counterdiabatic — marginal; letter-pass, spirit-fail

The derivation is clean: a gauge transform shows a grating phase gradient
φ′(x) ≡ local detuning, so Berry's counter-term lives in the **quadrature
channel = per-tooth position shifts** — genuinely distinct from the spent
width-profile route (2A), and trivially binary-etch realizable. The uniform
gap-shift pair [+20,+20] is the crudest member of this family.

But the constrained solve in the validated mode-B kernel says the channel is
already saturated at the stack: optimal 14-tooth translation profile
(max |s| = 8.6 nm, 76% orthogonal content to the uniform pair) predicts
**−2.9% in-cone — statistically identical to the uniform pair's −2.8%** in the
same kernel, saturating by scale×2 and reversing by ×3. Kernel caveat: it
under-responds on its own calibration anchor (−2.8% predicted vs −17.3%
measured for the [30,30] step), so absolute numbers are soft; the relative
"no headroom beyond the existing pair knob" is the honest reading.
**Recommendation: LOW priority — at most a ~10-row falsifier (profile ×
±scale) piggybacked on a big array, or skip.**

---

## Proposed next step (needs your pick — NOTHING dispatched)

Batch-1 candidate (one combined `--option3` array, accurate mesh, converged
box, width tracked, control + jitter partner per study, chunked to QOS 100/4):

1. **FW-BIC side-cavity scan (4.1b)** — TM + TE: {gap → κ} × {partner detuning
   → D} around the CMT point, ~30 rows/pol + single-resonance check rows.
2. **Vertical anti-phase alternation (4.1c)** — {dW₂ amplitude} × {phase/tooth
   offset}, ~12 rows (cheapest route to the vertical 38%).
3. **Forward-Huygens scatterers (4.2)** — radius bracket around TM 250 nm /
   TE 260 nm at the 4.4 best site (x₀≈0.25, y₀≈0.65) + legacy 810 nm site +
   matched plain-cylinder controls, ~20 rows.
4. (optional, cheap) ±40 µm 1D line export row to fix the 4.4 near-axial
   resolution; ~10-row counterdiabatic falsifier if you want 4.3 buried
   properly.
