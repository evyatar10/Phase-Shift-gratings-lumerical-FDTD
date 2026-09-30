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
| in-core oxide comb, full phase x period x count grid (2026-09-14/15, Athena 148812 / 149355 / 149982, r 80 / y 250 / 31 holes) | same circle SHAPE as the SiN comb (best 270°, worst 90°; Λ536 fit swing 0.165 vs SiN 0.026). At 270° T rises monotonically as Λ falls and PEAKS at **Λ 524: T 0.9280** (520: 0.9173, 527: 0.9000, 515: 0.831, 510: 0.745) — ABOVE ctrl 0.8851 and above the best SiN comb (0.8967 @530). CAVEAT (§2 sanity): the in-core rows are NOT width-neutral — resonance shifts to 1555.6-1557 nm and the spatial mode width moves 15.5 → 23.0 µm (524) / 24.8 (527) / 12.0 (510); resonant loss 0.11 → 0.067. Candidate, single-family, unconverged vs the fixed-width spec. Count does not recover at Λ531 (3/9/31 holes all ~0.85-0.86). r 110 at the same point (job 150391): T 0.8869, λ 1554.17, width 28.2 µm, loss 0.101 — bigger holes push the width/λ lever further but T falls back to ctrl (overdrive). r 50 at the same point (job 150429, one mesh cell at dx 50 — staircased, candidate until checked at accurate mesh): T 0.9244, λ 1557.80, spectral FWHM 1.207 (Q 1290), width 18.5 µm, loss 0.073 — keeps ~90% of the r 80 T gain while the width penalty drops from +7.5 to +3.0 µm and the λ shift from −2.3 to −0.8 nm; radius is a monotone width/λ lever (r50 18.5 / r80 23.0 / r110 28.2 µm) with T peaking near r 50-80. Radius series completed with r 40 / r 30 (job 150458, sub-cell at dx 50, trend only): r30 T 0.9008 / λ 1558.41 / Q 1320 / w 16.40 µm / loss 0.096; r40 0.9113 / 1558.17 / 1309 / 17.23 / 0.086. So along r = 30→50→80 the width closes smoothly toward the ctrl (15.5) while T falls smoothly toward the ctrl (0.885): ΔT/Δwidth is ~constant ≈ +0.018 per µm of extra width, i.e. the in-core comb buys T by the SAME lever that widens the mode — there is no radius at which the T gain survives at ctrl width. EQUAL-WIDTH TEST (job 150488, `scat_x8_incore_r50_c477`): r 50 holes + corr raised 400→477 (q3db knob line) landed the width at 15.39 µm (target 15.5, knob good to 1%) but T 0.7922 / λ 1556.24 / Q_L 2102 / loss 0.193 / Q_i 19.1k vs plain ctrl 0.8851 / 1327 / 0.111 / Q_i 22.4k at 15.53 µm. Decomposition: Q_c 1410→2362 (+67%, the deeper corrugation at fixed N=80 strengthens the mirrors — the pre-registered null band 0.87-0.90 missed because it assumed T tracks width only) and Q_i −15% (width-blind, the clean verdict): at matched width the hole comb RADIATES MORE than the plain device. CLOSED NEGATIVE: the in-core T gain was entirely the corr→width lever; the envelope-shape effect of the holes at fixed width is negative. r 50 phase circle at Λ524 (job 150504 + X6): 0° 0.8285 / 90° 0.7846 / 180° 0.8293 / 270° 0.9244, fit 0.842 + 0.070·cos(φ−270°) — same optimum phase as r 80 (0.612 + 0.165) and SiN (0.866 + 0.026); amplitude ∝ hole dose, mean level falls with dose. ON-AXIS SINGLE HOLE (y 0, stages X10-X12, jobs 151320/151333/90593/151353, r 30-110 at Λ524/270°): T 0.898/0.907/0.915/0.921/0.923/0.903, width 16.2/16.8/17.4/18.1/20.1/22.8 µm for r 30/40/50/60/80/110 — the same T-vs-width line as the pair (±0.002 at matched width), just a lower dose per radius (axis r 50 ≈ pair r 42). EQUAL-WIDTH TEST #2 (job 151476, `scat_x17_incore_axis_eqwidth`): axis r 60 + corr 465 → width 15.52 (target 15.53, knob exact), T 0.8198, Q_L 1920, Q_c 2121, Q_i 20.3k (−9% vs ctrl 22.4k); axis r 80 + corr 503 → width 15.79, T 0.7490, Q_L 2335, Q_c 2698, Q_i 17.3k (−23%). Both inside the pre-registered bands. q3db-engine extend-mode estimates of the −3 dB device at this width (EXPECTED, single-row anchors, ±10% Q_L): plain corr 400 N≈122 Q_L 6627; axis r 60/c465 N≈107 Q_L 6117 (−8%); pair r 50/c477 N≈103 Q_L 5715 (−14%); axis r 80/c503 N≈97 Q_L 5031 (−24%). Series completed with r 30 @ corr 418 (w 15.34, T 0.8622, Q_L 1512, Q_i 21.2k, −6%) and r 40 @ 433 (w 15.36, T 0.8509, Q_L 1635, Q_i 21.1k, −6%) (job 151664, `scat_x20_incore_axis_eqwidth_r30_40`); engine −3 dB: 6272 (N≈116) / 6258 (N≈113) vs plain 6629 (−5%/−6%). Every in-core hole comb LOWERS the equal-width Q3dB Q (the SiN outer comb raises it +16-30%); the smallest doses lose the least (−5%) and none crosses zero. Partial on-axis phase/period scans (X13 Λ510/515/520/527 @270°, X16 3-hole Λ531 @0°) were cancelled by the user mid-way; the finished rows remain on the servers, not fetched. AXIS r 60 FULL PROGRAM (jobs 151509 Athena + 90726 IGUM, `scat_x18_axis_r60_period_c536` / `scat_x19_axis_r60_phase524`): 270° period scan T 0.837/0.868/0.900/0.921/0.927/0.927/0.919/0.913/0.898/0.876 at Λ510/515/520/524/527/530/534/536/540/545 (optimum 527-530, broader and later than the pair's 524; width 13.6→18.7→16.1 µm tracks T); phase circles Λ524: 0.822/0.772/0.823/0.921, Λ536: 0.811/0.705/0.806/0.913 (0/90/180/270) — same 270° optimum as every comb. REDONE AT THE r 60 OPTIMUM Λ527 (jobs 151686 Athena + 90966 IGUM, `scat_x21_axis_l527_athena` / `scat_x22_axis_l527_igum`): phase circle r 60: 0.823/0.741/0.823/0.927 (0/90/180/270), width 15.5/13.6/15.7/18.6 µm; radius series @270°: r30 0.903/16.3, r40 0.913/17.0, r50 0.923/17.7, r60 0.927/18.6, r80 0.914/21.1, r110 0.856/24.8 (T / µm) — T peaks at r 60 at this period. r 50 phase circle @Λ527 (job 151719): 0.851/0.792/0.852/0.923. EQUAL-WIDTH RESCALES AT Λ527 (jobs 151719 Athena / 91008 IGUM, `scat_x23_axis_l527_r50phase_r30eq` / `scat_x24_axis_l527_r50eq`): r 30 @ corr 421 → w 15.38, T 0.8642, Q_L 1538, Q_c 1654, Q_i 21.8k (−2.6%), engine −3 dB Q 6456 @ N≈116 (−2.6%); r 50 @ corr 456 → w 15.52, T 0.8394, Q_L 1850, Q_c 2019, Q_i 22.1k (−1.5%), engine −3 dB Q 6625 @ N≈110 (0.0%). At the r-60 optimum period the equal-width penalty shrinks to the noise level (≤3%, inside the engine's ±10% band) — NEUTRAL, still no gain; the pair/Λ524 rescales were −6…−24%. Figures `matlab_plotting/studies/plot_axis_r60_summary.m` → results_from_athena/scat_x21_axis_l527_athena/. Plots `matlab_plotting/studies/plot_incore_circle{,_fit}.m` | `scat_x2_incore_circle`, `scat_x3_incore_lamscan`, `scat_x4_incore_below530`, `scat_x5_incore_r110`, `scat_x6_incore_r50`, `scat_x7_incore_r40_r30`, `scat_x8_incore_r50_c477`, `scat_x9_incore_r50_phase`, `scat_x10_incore_r50_axis`, `scat_x11_incore_axis_r30_40_60`, `scat_x12_incore_axis_r80_110`, `scat_x17_incore_axis_eqwidth`, `scat_x18_axis_r60_period_c536`, `scat_x19_axis_r60_phase524`, `scat_x20_incore_axis_eqwidth_r30_40` (+ cancelled `scat_x13..x16`) |
| air (oxide-index) comb | mechanism study only; π-flip confirmed; device stays SiN | `scat_air_comb` |
| the 2-pillar pair | **permanently dropped by user order** — "pillars" always means this periodic row | project rule |

## 7. Where the comb lives now

The comb is a **first-class part of the corr-325 adjoint inverse-design campaign**: of
the 191 free parameters, **115 are the comb** (57 radii + 57 x-positions + the standoff
d), so the optimizer may break the uniform lattice entirely. See
`runners/lumopt2_design/THEORY.md`. Everything above is the *hand-designed* comb that
seeds it.

## 7b. ★2026-09-11 — physics-first rethink: read `docs/comb_physics_rethink_2026-09-11.md`

A validated k-space interference model (`python_tools/comb_kspace_model.py`, calibrated on
13 stored rows, 27/37 blind cladding-comb rows within 2× the noise floor) re-reads this
whole file. What changes: (1) the "needle at ux = 0.98" of §1/§4 was the far-field
monitor's clipping edge — the leak piles up at the horizon and the correct aim is the
cutoff Λ_c = λ/(n_eff + n_clad) = 530.6 nm, which is why the T-plateau sits at 530–532 and
not at 536; (2) the r²-vs-r⁴ budget of §5.5 is ordinary two-channel interference (own
emission ∝ a², interference ∝ a) — the comb was at its amplitude optimum, ceiling
η²P_n = S²/4P_c = +0.011 from the stage-R circle alone; (3) the part of the leak the comb
interferes with behaves as top-going, so a second row at any standoff is amplitude only —
§6's "second row" verdict is confirmed with a mechanism, and an azimuthally designed second
row (tested as a hypothesis in the rethink) is withdrawn; (4) within the validated range
the single row is at its ceiling (period, phase, post count); the only untested axis is
the post amplitude at the cutoff (no stored row has r > 110 at Λ = 531, d = 1.8); (5) the
comb's value shrinks with any envelope smoothing and it belongs outside the inverse
design; (6) at the 20 µm spec the stored apodized corr-400 N=150 rows (Q_i 217k; ≈620k
with the full-z trench) are 4–10× the comb lock's Q_i — Quan & Lončar's cusp argument in
the group's own numbers.

**★2026-09-12 — the OFF-CENTRE comb, MEASURED (Athena job 146364, 9/9; doc §6c).** The
comb's phase-independent cost is set by its CENTRE along the guide, not by δx: the Λ-536
circle displaced to ±12 µm has pedestal −0.009 / −0.002 (centred: +0.184 in Δγ/γ) and its
optimum phase flips 270° → 90° (half-period π/(β − k_c) = 12.2 µm). Best single displaced
comb T 0.8967 (−12 µm) / 0.8959 (+12 µm) vs centred 0.8928 and ctrl 0.8851 — a single comb is
at its ceiling either way, because the swing fell with the pedestal (carrier ×0.62 at 12 µm).
The two sides are mirror images (no standing-wave asymmetry). Every earlier second-comb
attempt (§6, stage T rows at d, d+Λ…) shared the first comb's centre and could only add
amplitude. The k-space model refit on all 16 circle rows (rms 0.0034) predicts a PAIR — one
comb per side at its own phase — is nearly additive (+0.0285 / +0.0254; measured singles sum
to +0.0225) and a THIRD comb adds ≤ +0.001. Dispatched as `scat_offcentre2.py` (Athena job
146419) and MEASURED: **pair T 0.9040 (+0.0189), triple 0.9031 (+0.0180)** — a second comb on
its own centre beats the best single (+0.0115) by 4× the floor; the third adds nothing. THE
PATTERN: one 31-post comb per side at ±12 µm (the end-scattering half-period), each at 90° of
Λ 536, r 110, d 1.8 — never two combs on one centre. Loss 0.111 → 0.093, R and λ unchanged.
Doc §6c–6d, fig8. Far-field monitors at 2.0 wls verified harmless for T (control 0.8864 vs
stored 0.8851) — but keep the TOP monitor ≥ ~3 µm (evanescent-tail truncation ripple).
**★ROUND 2–3 MEASURED (jobs 146553 / 146557 / 146564, doc §6e–6f):** pair at the cutoff period Λ 531
(model phases +12 @60°, −12 @120°) **T 0.9070** (+0.0219); pair on **W1050 cavity T 0.9374**
(+0.0156, single comb there 0.9310); rod posts 140×270 = circles (0.9038); pair on apod-10 **loses**
(0.9708, −0.0062: no needle left to patch — comb and apodization are either/or); **one centred
61-post comb at r 110, Λ 531, 270° T 0.9064** = the pair (the comb's cost follows its ENDS, sign
period 12.2 µm; the stored N61 row at r 78, +0.0137, was amplitude-starved by the old Σr² rule).
Model (16-row refit, +0.004 optimistic): third comb, fan combs, extra rows, curves/chirps/clusters
(free-form optimizer) all tie or lose. Design rule: ends at ±16 µm (or two 31-post combs at ±12),
Λ = Λ_c, r 110, d 1.8. **Round 4 (job 146614): two 61-post combs at ±17 µm (244 posts, ±33 µm)
0.9057 / 0.9040 = TIE with the 31-pair — saturated by ~61 posts; the model's +0.007 was its largest
miss (optimism grows with post count: +0.004 @31, +0.008 @62, +0.018 @122 — rank only, never
extrapolate to bigger arrays). PROGRAM CLOSED on this device.**
**★Q3DB TRANSFER (job 146639, doc §6h): corr-325 N165 at comb_q3db numerics — one 61-post r-110
comb T 0.5659 (+0.0753), the r-110 pair (31 @ +12/60°, 31 @ −12/120°) T 0.5704 (+0.0798); both far
above the stored r-80 57-post comb (0.5361); pair leads by +0.0045 (candidate, inside the ~0.005
working floor at T ≈ 0.5). predict-q3db extend: pair N* = 172 → Q_L 18155 / width 19.77 µm; single
N* = 171 → Q_L 17620. Old comb lock 16203 (N169); trench 18777. **CONFIRMED (job 146681): pair N172 T 0.5003 (−3.01 dB)
Q 18093 w 19.76 µm; single N171 T 0.5060 (−2.96 dB) Q 17557 w 19.79 µm — engine misses ≤ 0.4 % on Q.
NEW Q3DB DEVICE: corr 325, N 172, W800, comb pair r 110 / d 1.8 / Λ 531: 31 posts at +12 µm (δx 88.5 nm)
+ 31 posts at −12 µm (δx 177 nm), mirrored ±y. Runner runners/metal_mirror/comb_q3db_lock.py.**

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
