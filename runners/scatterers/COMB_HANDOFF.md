# COMB HANDOFF — the cladding post comb ("the circles") on the pi-shift Bragg grating

Written 2026-08-27. Companion to `runners/lumopt2_design/HANDOFF.md` (inverse-design
state) and its `THEORY.md` (method). **This file is SELF-CONTAINED**: paste it into a
fresh Claude chat and it has everything needed to build a presentation — device,
mechanism, conventions, every measured number with its source file, what is closed,
what is open.

Provenance labels, per the project's honesty rule: **MEASURED** = read from a named
`.mat` result file · **DERIVED** = computed from measured values · **EXPECTED** =
model/theory/estimate. Every table below is MEASURED unless marked otherwise; all
values were re-read from the `.mat` files on 2026-08-27.

---

## 1. The device, in one paragraph

A **pi-shift Bragg grating**: a SiN strip waveguide (n_core 1.97) in oxide cladding
(n_clad 1.444), core height 350 nm, average width 800 nm, sidewall corrugation of depth
`corr`, pitch 516.83 nm, and a **half-period (pi) phase slip at the centre**. The slip
creates a defect state inside the stop band — a resonance at ~1559 nm whose mode is
spatially extended along the guide (~15–20 µm FWHM). Two families matter: **corr = 400 nm
with N = 80 periods/side** (the loss-physics workhorse) and **corr = 325 nm with
N = 165–169** (the "q3db" operating point: peak transmission at −3 dB with a 20 µm mode,
which is the acoustic-detector spec). Polarization is **TM** unless stated.

The performance limit is **radiation**: the resonance leaks out of plane, and that leak
is dominated by a **grazing lobe near ux ≈ 0.98** — nicknamed the **"needle"**. Killing
or recycling the needle is what the comb is for.

## 2. What the comb IS (geometry)

Two rows of small **SiN cylinders ("posts", "the circles")** in the oxide cladding — one
row at `+d`, its mirror at `−d` — running parallel to the guide:

| symbol | meaning | winner value (corr-400) | q3db value (corr-325) |
|---|---|---|---|
| `Λ` | comb period along x | 531 nm | 531 nm |
| `δx` | rigid shift of the whole comb along x | 398 nm (= 270°) | 401 nm (= 270°) |
| `r` | post radius | 80–110 nm (broad plateau) | 80 nm |
| `d` | standoff, guide axis → post centre | 1.8 µm | 1.9 µm |
| `h` | post height | 350 nm = core height (**single litho**) | 350 nm |
| `n` | posts per row | 41–53 (best), 31 (early) | 57 |

Note `Λ = 531 nm ≠ grating pitch 516.83 nm`. The comb is **not** matched to the grating;
it is matched to the *radiation* it must cancel (§4). The posts sit in the cladding,
~1.3 µm clear of the tooth edges — they never touch the waveguide.

## 3. ★THE PHASE — "270°, relative to WHAT?"

**Definition (exact, as used in every runner and every table here):**

> **φ = 360° × δx / Λ**, where `δx` is the rigid translation of the entire comb lattice
> along the propagation axis, measured from **x = 0 = the pi-shift defect at the centre
> of the cavity**. Post positions are `x_k = k·Λ + δx`, k = −n_half … +n_half.

The reference point is **the cavity centre**, and the unit of phase is **one comb period
Λ** — not the grating pitch, not the optical wavelength:

- **φ = 0°** ⇒ a post sits exactly on the defect axis (x = 0).
- **φ = 270°** ⇒ the lattice is shifted by 3/4 of a comb period: δx = 0.75 Λ
  (531 × 0.75 = 398 nm; the q3db comb uses 401 nm = 271.9°, rounded to "270°").

**Where exactly is x = 0? MEASURED from the built scene** (`smoke_0.fsp` of job 137831,
corr-325 N=100; segment coordinates read back from the .fsp on 2026-08-27):

| object | x_min (nm) | x_max (nm) | length (nm) | width (nm) |
|---|---|---|---|---|
| **cavity segment** | **−129.21** | **+129.21** | 258.41 | 800.0 (avg) |
| L_wide_1 (left neighbour) | −387.62 | −129.21 | 258.41 | 962.5 (**wide**) |
| R_narrow_1 (right neighbour) | +129.21 | +387.62 | 258.41 | 637.5 (**narrow**) |

So **x = 0 is the exact CENTRE of the cavity segment**, which spans ±pitch/4 = ±129.21 nm
— not an interface. Two consequences that matter when quoting the phase:

- **Re-referencing costs a quadrant.** Measured from the cavity→narrow interface
  (+129.21 nm) instead of the centre, every phase shifts by 360° × 129.21/531 =
  **87.6°**: "270° from the cavity centre" = "182° from the cavity/narrow edge" =
  "358° from the wide/cavity edge". Always state the reference as *the centre of the
  cavity segment*.
- **The device is NOT mirror-symmetric about x = 0**: the cavity has a WIDE tooth on its
  left and a NARROW section on its right (that half-period slip IS the π shift). So no
  symmetry argument ties φ to −φ, and the measured T(90°) ≠ T(270°) (§5.1) needs no
  special explanation.
- Side note found in the same check: the YZ cross-section, side and top monitors are
  placed at `x = cavity_length/2` = +129.21 nm (code comment says "centered on
  phase-shift defect") — that is the cavity's right EDGE, a quarter pitch off centre.
  Negligible for the ±30 µm side/top far-field spans; it does mean the YZ slice is taken
  at the cavity/narrow boundary.

**What it means physically.** Λ is chosen so consecutive posts re-radiate **in phase**
into the needle direction (§4). Translating the lattice by δx therefore does not change
the *shape* of the comb's radiated beam at all — it only advances that beam's **phase**
by 2π·δx/Λ, while the grating's own leakage does not move. Hence:

> **φ is the relative phase between the comb-radiated beam and the device's own
> radiation lobe, in the far field.** It is an interference phase; δx is the knob that
> turns it. One full turn of φ = one comb period of translation.

**φ = 270° is EMPIRICAL, not analytic** — it is simply where the measured interference
is destructive. There is no reason for the extremum to land on the geometric origin: the
constant offset between the comb's re-radiation phase and the device's leak phase at the
cavity is set by the device itself, and the measurement puts the null at 3Λ/4. The
measured phase circle (§5.1) is a clean sinusoid with its **maximum near 90° and its
minimum near 270°** — exactly a two-channel interference.

**Independent confirmation:** the TE comb study (different polarization, corrugation
300 nm, different comb period Λ = 590 nm) peaks at δx = 443 nm = **270.3°**. Same
convention, same answer — MEASURED, `results_from_igum/scat_te_comb/`.

## 4. The mechanism (why a comb at all)

1. The resonance leaks out of plane; most of the leak sits in a narrow grazing
   **"needle"** lobe (|ux| ≈ 0.98).
2. A periodic row of cladding posts is a **grating for the guided carrier**: it supplies
   a reciprocal-lattice vector G = 2π/Λ that **out-couples** guided light into a beam
   whose angle is set by Λ. This was found by accident — the stage-O full-depth comb
   (Λ = 551 nm) radiated a **new, +10 dB lobe** at ux −0.925 and collapsed T from 0.873
   to 0.687 (MEASURED, `scat_o_comb1800`).
3. **Retune Λ so that new beam lands ON the needle**, then use δx to put it in
   anti-phase → the two channels **destructively interfere**. This is a
   Friedrich–Wintgen-style two-channel cancellation (same math as magic-width lateral-
   leakage cancellation in SOI); the novelty here is that the phase is controlled by a
   **rigid translation δx** of a separate structure.
4. Zero-GPU calibrated design model (`python_tools/antineedle_comb_design.py`, figure
   `docs/antineedle_comb_design.png`): n_eff from the measured beam angle = 1.4936; a
   width-matched comb of ~17 µm reaches **~78 % needle-power cancellation** at optimal
   phase, while a full-length 83 µm comb caps at ~20 % — its beam becomes too narrow in
   angle to overlap the needle. (EXPECTED values; FDTD confirmations in §5.)

## 5. The measured record

### 5.1 The phase circle — the smoking gun

TM corr-400, N = 80/side, Λ = 545, r = 110, d = 1.8 µm, 31 posts, box y = 16 µm,
20 nm / 1501 pts, optimization mesh. Source:
`results_from_athena/scat_p_antineedle/results/*.mat` (job 129989, 9/9 completed).
Control (no comb) at identical numerics: **T = 0.8851** (recorded in the study docs from
job 123563; that file was not re-opened for this handoff).

| δx (nm) | φ | peak T | needle power vs control |
|---|---|---|---|
| 0 | 0° | 0.8694 | ×1.29 |
| 136 | 90° | 0.8586 | ×2.19 (worst) |
| 273 | 180° | 0.8689 | ×1.35 |
| **409** | **270°** | **0.8797** | **×0.449 (−55 %)** |

T column MEASURED; the needle column is recorded from the far-field reduction of the
same job (not re-derived here). **The needle can be more than halved by translating the
comb — pure phase control.** λ pull is only +24–37 pm and reflection is flat (+0.0004),
so the comb is not acting as a parasitic mirror.

### 5.2 The aim (Λ) scan, at φ = 0

Same family, r = 110, δx = 0: T = 0.8664 (Λ 539) / 0.8675 (542) / 0.8694 (545) / 0.8714
(548) / 0.8730 (551) — monotone, all on the constructive side. **Aim conclusions must be
read off the phase circle, never off a δx = 0 cut.**

### 5.3 The winner at corr-400, and the length axis

Λ = 531 nm, δx = 398 nm (270°), d = 1.8 µm, h = 350 nm. Sources: `scat_s_refine`,
`scat_t_confirm`, `scat_y_polish`.

| posts | r (nm) | peak T | Δ vs control 0.8851 |
|---|---|---|---|
| 31 | 110 | 0.8966 | +0.0115 |
| **41** | **96** | **0.8999** | **+0.0148** |
| **47** | **89** | **0.9001** | **+0.0150** |
| 53 | 84 | 0.8999 | +0.0148 |
| 61 | 78 | 0.8988 | +0.0137 |
| 151 (full device) | 80 (d 2.28) | 0.8920 | +0.0069 |

Radii are amplitude-matched (r ∝ 1/√n) so every row drives the same total amplitude.
**41–53 posts is the plateau; the full-length comb is clearly worse** — confirming the
design model's beam-width argument (§4.4). The radius plateau at 31 posts is broad:
r = 85 / 92 / 100 / 110 → T = 0.8932 / 0.8951 / 0.8936 / 0.8966, i.e. spread at the
numerical floor (the dx = 50 nm mesh jitter floor in this program is ±0.0018).

### 5.4 Standoff, height, and the two fab routes

- **d is nearly degenerate** once r is re-matched: d = 1.5 µm / r 82 → T 0.8974;
  d = 1.8 / r ≈ 96 → 0.8999; d = 2.1 / r 147 → 0.8911 (`scat_w_dscan`, `scat_y_polish`).
- **Core-height posts (h = 350 nm) = single litho** — same etch step as the teeth.
- **Deep-etch "flush" posts** (bottom at −3.975 µm) also work if the radius is reduced to
  compensate: r = 70 flush → T 0.8990 vs its own z-asymmetric control 0.8864
  (**+0.0126**), while r = 110 flush is catastrophic (0.8341) — too much amplitude plus a
  drain path (`scat_u_flushcomb`). **Two equivalent fab routes; single-litho preferred.**

### 5.5 What limits it: the r² vs r⁴ budget

Each post does two things: it **coherently** re-radiates the anti-needle beam (amplitude
∝ r²) and it **incoherently** scatters the carrier into all other angles (parasitic loss
∝ r⁴ — measured 0.0039 @ r80 → 0.0145 @ r110, ratio 3.7 ≈ (110/80)⁴). Optimising the
amplitude of that trade at corr-400 gives a ceiling of only **≈ +0.001 in T**, at or
below the noise floor. **DERIVED, recorded 2026-08-10: the comb of circular posts is
closed as a pure T-lever at corr-400.** Its value lies elsewhere — next section.

### 5.6 ★The headline: the q3db operating point (corr-325)

The number to present. Device: TM corr-325, W800, h350, box y = 8 µm, window 20 nm @
1559.5 nm, 4001 points, optimization mesh. Comb: Λ 531 / δx 401 (270°) / r 80 / d 1.9 µm
/ 57 posts / h 350. Source: `results_from_athena/comb_q3db/results/` (jobs 130458 +
130548) — all MEASURED.

| row | N | T | dB | λ (nm) | Q | mode FWHM |
|---|---|---|---|---|---|---|
| control, no comb | 165 | 0.4906 | −3.09 | 1559.001 | 13 930 | 19.97 µm |
| comb **270°** | 165 | 0.5361 | −2.71 | 1559.011 | 14 584 | 19.90 µm |
| comb 90° (sign check) | 165 | 0.4371 | — | 1559.016 | 13 143 | 20.05 µm |
| comb Λ = 536 (aim hedge) | 165 | 0.5283 | — | 1559.016 | 14 476 | 19.96 µm |
| comb 270° | 167 | 0.5160 | −2.874 | 1559.011 | 15 352 | 19.90 µm |
| comb 270° | 168 | 0.5059 | −2.960 | 1559.011 | 15 761 | 19.91 µm |
| **comb 270° — THE LOCK** | **169** | **0.4961** | **−3.044** | **1559.011** | **16 203** | **19.91 µm** |

- The comb buys **+0.0455 T (+0.385 dB)** at fixed N = 165; that surplus is spent by
  lengthening the device to N = 169, where it returns **Q = 16 203 vs 13 930** —
  **+16.3 % Q at exactly the same −3 dB spec and the same 20 µm mode** (DERIVED from the
  table; the −3 dB crossing is N ≈ 168.5, slope −0.086 dB/period).
- The **90° row loses** (Q 13 143 < control) — the sign check the mechanism demands.
- λ is pinned at 1559.011 on every comb row: the comb does **not** pull the resonance.

**Benchmark at −3 dB, each family at its own lock** (MEASURED across studies):

| decoration | lock | Q | vs control |
|---|---|---|---|
| full-z air trench | N = 170 | 18 777 | +34.8 % |
| flush air trench | N = 168 | 16 942 | +21.6 % |
| **post comb (this work)** | **N = 169** | **16 203** | **+16.3 %** |
| none (control) | N = 165 | 13 930 | — |
| TE, corr 250 | N = 166 | 12 903 | −7 % |

**Honest ranking: the comb is third on Q.** Its differentiators, all MEASURED at the
operating point: **zero mode-width cost** (19.91 µm vs 19.97 control, spec 20 µm),
**single-litho fabrication** (no deep etch anywhere) and **no resonance pull** (+10 pm).
The trenches win on Q but require a deep etch.

## 6. Closed axes — do NOT re-open (each already cost GPU time)

| axis | verdict | evidence |
|---|---|---|
| chirp / non-uniform spacing | ≡ a period shift; quadratic residual < 5 % field = sub-floor | zero-GPU, measured leak phase slope 0.09 rad/µm |
| envelope apodization of radii | null: T 0.8944 vs 0.8966 uniform | `scat_t_confirm` row 7 |
| second row / 2D lattice | no gain — rows overshoot or merely re-equal one row | `scat_t_confirm` rows 4–6 (0.8938–0.8950) |
| r > 110 | degrades (r 400 → −0.24 T) | stages W/Y |
| full-device-length comb | worse than 41–53 posts | §5.3 |
| comb on an apodized device | does **not** transfer: T 0.9723 vs apod-10 control 0.9770 | `scat_v_apodcomb` |
| in-core oxide holes (inverted posts) | harmful: 0.8460 / 0.8654 / 0.5438 | `scat_x_incore` |
| air (oxide-index) comb | mechanism study only; π-flip confirmed; device stays SiN | `scat_air_comb` |
| the 2-pillar pair | **permanently dropped by user order** — "pillars" always means this periodic row | project rule |

## 7. Where the comb lives now

The comb is a **first-class part of the corr-325 adjoint inverse-design campaign**: of
the 191 free parameters, **115 are the comb** (57 radii + 57 x-positions + the standoff
d), so the optimizer may break the uniform lattice entirely. See
`runners/lumopt2_design/THEORY.md`. Everything above is the *hand-designed* comb that
seeds it.

## 8. Open / in flight

- **★CLOSED 2026-08-27 — SHAPE DOES NOT MATTER, AREA DOES (Athena job 137831, 4/4
  COMPLETED, exit 0).** Question: do the posts have to be circles? Same 57 sites, same
  Λ / δx / d / h, corr-325 at the N = 100 surrogate, campaign box 6.8/6.8 µm. All
  MEASURED, `results_from_athena/scat_rect_comb/results/`; circle control reused, not
  re-run (`tm_comb_box_c325`, identical numerics).

  | post | area vs r80 | T | ΔT vs circle r80 | λ (nm) | mode |
  |---|---|---|---|---|---|
  | circle r 80 (control) | 1.00× | 0.92079 | — | 1559.011 | 19.17 µm |
  | rect 142×142 (equal area) | 1.00× | 0.92012 | −0.0007 | 1559.011 | 19.18 µm |
  | rect 100×200 (equal area, elongated across the guide) | 0.99× | 0.92105 | +0.0003 | 1559.011 | 19.17 µm |
  | rect 160×160 (same bounding box) | 1.27× | 0.92335 | +0.0026 | 1559.016 | 19.15 µm |
  | circle r 90.3 (equal area to the 160 square) | 1.27× | 0.92302 | +0.0022 | 1559.011 | 19.16 µm |

  **Verdict:** an equal-area square ties the circle (−0.0007, inside the ±0.0018 jitter
  floor) and a 1:2 aspect ratio at the same area changes nothing (+0.0003); the 160 nm
  square and the equal-area circle r 90.3 agree to 0.0004, so the +0.0026 of the bigger
  square is bought by its 27 % extra area, not by its corners. **Fab implication: draw
  whatever shape is convenient, but match the AREA** — a square of side r·√π ≈ 1.77 r
  reproduces a circle of radius r; a square of side 2r is a 27 % larger post.
  Caveat: the +0.002x gains sit just above the noise floor with no jitter twin in this
  sweep — CANDIDATE, not confirmed. The null result (shape-independence) is a difference
  *inside* the floor, measured twice, and is unaffected.
  Runner: `runners/scatterers/scat_rect_comb.py`.

- **Smooth sinusoidal width modulation** instead of discrete posts — the only idea that
  could beat the r² vs r⁴ budget (moves Fourier weight out of the broadband parasitic
  channel into the G-line). Never dispatched.
- **Comb + trench** combination — untested.

## 9. Figures that already exist (for the deck)

| file | shows |
|---|---|
| `results_from_athena/comb_q3db/comb_q3db_benchmark.png` | T(dB) and Q vs N, all four families — **the money figure** |
| `results_from_athena/comb_q3db/comb_q3db_N169_transmission.png` | the locked device's spectrum |
| `results_from_athena/comb_q3db/comb_q3db_N169_mode_width.png` | its 19.91 µm mode |
| `results_from_athena/scat_rect_comb/comb_phase_convention.png` | **what the 270° IS**: the comb drawn against the phase-0 reference lattice at the defect + the measured phase dial |
| `results_from_athena/q20um_3db_benchmark/comb_phase_scan.png` | peak T vs δx (phase) at fixed Λ = 536 nm — the oscillation |
| `results_from_athena/q20um_3db_benchmark/comb_period_scan.png` | peak T vs Λ at 270° vs 0° — the aim curve; at the wrong phase no period helps |
| `results_from_athena/scat_p_antineedle/scat_p_antineedle.png` | the phase circle + Λ scan |
| `docs/antineedle_comb_design.png` | the zero-GPU design model (cancellation vs Λ and length) |

MATLAB sources: `matlab_plotting/plot_comb_q3db.m`, `plot_scat_p_antineedle.m`,
`plot_antineedle_design.m`, `plot_comb_schematic.m`, `plot_comb_phase_scan.m`.

## 10. Suggested presentation arc

1. **Problem** — pi-shift Bragg grating at the acoustic-sensing spec (−3 dB peak, 20 µm
   mode); performance limited by out-of-plane radiation, dominated by a grazing needle.
2. **Observation** — a cladding post row out-couples the guided carrier into a beam
   (+10 dB lobe, first seen by accident in stage O).
3. **Idea** — retune Λ to aim that beam at the needle; translate by δx to set its phase
   → two-channel destructive interference.
4. **Proof** — the phase circle: needle ×2.19 at 90°, ×0.449 at 270°, sinusoidal;
   reproduced in TE at 270.3°. Phase = 360°·δx/Λ, measured from the cavity centre.
5. **Engineering** — 41–53 posts, amplitude-matched radii, d degenerate, single litho;
   and the r² vs r⁴ budget that caps the pure-T gain.
6. **Result at spec** — N = 169, −3.04 dB, **Q 16 203 (+16.3 %)**, mode 19.91 µm, λ
   unpulled; honest benchmark against the two trenches (higher Q, but deep etch).
7. **Next** — post shape (rect vs circle, in flight), smooth modulation, and the comb as
   115 free parameters inside the adjoint inverse design.
