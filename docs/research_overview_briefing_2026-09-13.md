# Research overview briefing — pi-shift Bragg grating program (2025-11 → 2026-09)

Purpose: the raw material for a SHORT, conclusion-first overview presentation
(research-proposal context). This is NOT a presentation. Hand it to Claude in
PowerPoint together with the "how to present" rules in §0.

Every number is tagged: MEASURED (read from a named result file this session),
DERIVED (computed from measured values), PROJECTED (extrapolated, never measured
at that point). Quote only MEASURED numbers as results; PROJECTED ones carry "~".

---

## 0. How to present (rules for whoever builds the slides)

**Style: assertion-evidence + pyramid.** Every slide title is a full-sentence
conclusion ("Air trenches raise Q by a third at equal insertion loss"), the body
is one figure or one small table that proves it, and details (if any) come after
the conclusion, never before. This is the Assertion-Evidence method (Alley, Penn
State; measured to improve audience recall, p<0.01) combined with Minto's Pyramid
Principle (answer first, then 2-3 supports, then evidence). Sources:
[assertion-evidence.com](https://www.assertion-evidence.com/),
[PSU self-study guide](https://www.assertion-evidence.com/guide.html),
[iBiology slide design](https://www.ibiology.org/professional-development/power-point-slide-design/),
[Pyramid Principle](https://managementconsulted.com/pyramid-principle/),
[Minto for slides](https://winningpresentations.com/pyramid-principle-presentations/).

**Vocabulary rules (user-mandated):**
- The device is a **pi-shift Bragg grating**. Never "champion", never "Q3dB".
- The benchmark is called **"the 20 µm / −3 dB operating point"**: a device whose
  spatial mode FWHM is 20 µm and whose peak transmission is −3 dB (T = 0.5),
  reached by lengthening the grating; the deliverable is the loaded Q there.
- "FWHM" / "mode width" = spatial width of the mode envelope (µm). Q is the
  loaded Q from the spectral linewidth.
- Numbers appear ONLY for the 20 µm / −3 dB devices (§2). Everything else is
  direction words (T up / width up / Q up / loss down), measured at equal period
  count.
- Physics: at most one equation per topic, only where §3 gives it.

**Length target:** ~12 slides. Suggested order = §1 (1 slide), §2 (1 slide, the
bar chart exists already), §3 TE (2), TE-vs-TM (1), TM levers (2), trench→circles
(2), inverse design (2), status table + next (1).

---

## 1. The one-slide bottom line

- Goal: a pi-shift Bragg grating for acousto-optic sensing with a FIXED ~20 µm
  mode, maximizing Q at a fixed insertion loss (−3 dB). Loss is radiation only
  (materials lossless), so every lever is a radiation-management lever.
- Everything that touches the TEETH trades loss against mode width: apodization,
  tooth shifts, tooth shapes all reduce loss AND widen the mode. Only two
  classes escape that trade: (a) changes at the cavity / new scatterers, and (b)
  the lab's overshoot apodization, which is a different regime altogether.
- TE and TM plain gratings reach the SAME Q at the 20 µm / −3 dB point (13k).
  TM radiates sideways in the plane, so it responds to lateral structures (air
  trench +35%, SiN circle rows +30%); TE radiates upward and does not.
- The inverse design (25 free teeth per side + cavity, mode width held fixed)
  is the biggest measured gain: Q 88.9k at the operating point, ~6× the plain
  grating. It is still below the lab's TE overshoot apodization (projected
  ~2×10⁵–3.4×10⁵) and is work in progress.
- Dead ends, all measured: reflectors of every kind (SiN DBR walls, metal
  mirrors, retro-combs), in-core hole lattices, BIC / Kerker scatterers, a
  second parallel cavity, shaped trenches, exotic innermost-tooth shapes.

---

## 2. The numbers: devices at the 20 µm / −3 dB operating point

All: SiN core n 1.97 / SiO₂ 1.444, height 350 nm, waveguide 800 nm, λ ≈ 1560 nm,
optimization mesh, each device lengthened until peak T = 0.5 with a 20 ± 1 µm
mode. Q = loaded Q. An existing chart of most rows:
`results_from_athena/q20um_3db_benchmark/q20um_3db_benchmark.png` (2026-08-26;
it lacks the newer circle-pair and full-trench rows).

| device | pol | corr / periods per side | peak T | Q | mode µm | tag |
|---|---|---|---|---|---|---|
| plain grating | TE | 250 nm / 166 | 0.492 | **12,903** | 20.5 | MEASURED |
| plain grating | TM | 325 nm / 165 | 0.491 | **13,930** | 20.0 | MEASURED |
| + SiN circle row (57 posts, r 80) | TM | 325 / 169 | 0.496 | **16,203** (+16%) | 19.9 | MEASURED |
| + air trench, half depth (single-litho top) | TM | 325 / 168 | 0.502 | **16,942** (+22%) | 19.7 | MEASURED |
| + single 61-post circle row (r 110) | TM | 325 / 171 | 0.506 | **17,557** (+26%) | 19.8 | MEASURED |
| + circle PAIR (31 + 31 posts, r 110, at ±12 µm) | TM | 325 / 172 | 0.500 | **18,093** (+30%) | 19.8 | MEASURED |
| + air trench, full depth (to substrate) | TM | 325 / 170 | 0.502 | **18,777** (+35%) | 19.6 | MEASURED |
| lab overshoot apodization (Itai, HH profile) | TM | his profile / 189 | 0.528 → 0.5 interp. | **~34,000–36,000** (2.6×) | 20.3 | MEASURED at N=189, interpolated to T=0.5 |
| inverse design, 25 teeth/side + circles | TM | ~358 mean / 220 | 0.499 | **88,868** (6.4×) | 19.9 | MEASURED (3 samples/linewidth caveat) |
| inverse design without circles | TM | — | — | ~76,000 | — | PROJECTED (scaled off the row above) |
| lab overshoot apodization (Itai, 60 teeth) | TE | his profile | — | **~2.0×10⁵ – 3.4×10⁵** (15×–26×), still rising with N | 19.7 | PROJECTED from MEASURED intrinsic Q via Q(−3 dB) = 0.293·Q_i: his scaled profile Q_i 6.7×10⁵ at N=195 → 2.0×10⁵; his re-optimized Nt60 profile Q_i 1.16×10⁶ at N=130 → 3.4×10⁵. The −3 dB point itself (N≈250) is too slow to simulate |

Sources: `memory/project_te_q3db_20um`, `project_trench_q3db_20um_closed`,
`project_trench_flush_top_study`, `runners/scatterers/COMB_HANDOFF.md`,
`project_comb_physics_rethink` (jobs 146639/146681), `runners/lumopt2_design/HANDOFF.md`
(N=220 row), `project_itai_hh_apodization`, `results_from_igum/itai_hh_nt60w20_summary.csv`,
`matlab_plotting/plot_q20um_3db_benchmark.m` (the ~76k and overshoot bars are
user-supplied projections, marked "~" in the script; the chart's old ~5.6e5
label is superseded: say "~2×10⁵–3.4×10⁵, projected").

Reading the table: the circle pair and the full trench are a tie (18.1k vs 18.8k,
inside the method's ±3% band); the circle pair needs no extra fab step. The
inverse design is a different tier. The lab's overshoot apodization is yet
another tier, and in TE only.

**The one equation for this slide:** at T = 0.5 the loaded Q is fixed by the
intrinsic (radiation) Q alone, Q(−3 dB) = (1 − √0.5)·Q_i ≈ 0.29·Q_i (measured to
~2% on four devices). So "Q at the operating point" is a pure measure of how
little the device radiates; period count is dictated, not free.

---

## 3. The story, direction by direction (bottom line first, then what it does)

### 3.1 TE — apodization
**Bottom line:** apodization is the strongest single loss lever and the most
expensive in mode width. Linear apodization: T up strongly, mode width up
strongly (e.g. 10 apodized periods per side: mode 15 → 20 µm), Q up. Tanh
apodization: same T for a tighter mode than linear (less widening). Number of
apodized periods saturates early (5 ≈ 10; 20–40 add nothing) and shifts the
resonance slightly. Physics: apodization removes the abrupt envelope corner at
the cavity that radiates as an antenna; far-field radiation drops 1–2 orders of
magnitude with 10 apodized periods. MEASURED (jobs 96506, 97137; meeting decks
2025-11 → 2026-03).

**Overshoot apodization (the lab's / Itai's model):** NOT a taper. The
corrugation first overshoots to ~2.4× the bulk depth ~25 periods from the
cavity, dips, and settles to the bulk. Only the profile AMPLITUDE sets the mode
width; the period count sets peak T. At ~20 µm it beats our plain grating by
~15× in TE and 2.6× in TM (intrinsic-Q ratio, MEASURED at N 130–195 TE / at the
crossing in TM). It is the strongest thing on the table and we did not invent it.

### 3.2 TE — tooth shifts ("radiation recycling")
**Bottom line:** shifting the innermost tooth pair outward (~100–140 nm, a
plateau) raises peak T from ~0.82 to ~0.87 with the mode width essentially
unchanged (15.4 → 15.9 µm), i.e. loss down at ~no width cost. Combined with a
one-tooth apodization: ~0.93, still at ~16 µm, vs ~0.96 at 20 µm for a full
10-period apodization. Q: not the headline; it rises modestly with T (the mode
barely changes). Mechanism (Lalanne-style): the shift co-tunes the phase of the
innermost reflection with the cavity round trip, arg(r) + k₀·n_eff·L_cav = 0,
so light that would leak at the cavity corner is redirected into the mode;
Poynting-vector maps show the flow turning inward above the inner teeth, and the
far field collapses from two side lobes to one near-axis lobe. **Correction to
your memory:** it does NOT reduce the mode width; it leaves it nearly constant
(slight widening). MEASURED (meetings 2026-04-05, 04-16).

**Why this only goes so far:** a k-space budget shows only ~15% of the radiated
power originates within ±1 tooth of the cavity (~30% within ±3); ~70% is
distributed along the arms. So innermost-teeth-only tricks (shift, exotic tooth
shapes) have a ceiling of ΔT ≈ +0.001…+0.006 beyond the shift itself. DERIVED
(`docs/theory_innermost_recycling_2026-07-08.md`).

### 3.3 TE — inverse design of the two innermost teeth (first round)
**Bottom line:** with only the two innermost teeth + cavity width free (5
parameters: two tooth depths, two shifts, cavity width) a particle-swarm search
reached peak T ≈ 0.972 at a 16.5 µm mode, the best T-per-width of anything at
that time (regular 0.85 → shift 0.92 → shift+1 apod 0.93 → PSO 0.97; linear
10-period apod 0.98 but at 20.4 µm). The adjoint gradient (Lumerical's lumopt
v1) did not converge at first (a gradient bug, later fixed to within ~15%), and
with only 5 parameters it was not needed; PSO was enough. Cost function was
peak T only; mode width was not controlled. The experiment (fabricated devices,
two pitches) confirmed the optimal shift range 100–130 nm; measured Q ran
1.3–1.5× above simulation. MEASURED (meeting 2026-05-12).

### 3.4 TE vs TM — direct comparison
**Bottom line (corrects your memory in one place):**
- Same length (80 periods/side, same corrugation): TM couples ~3× weaker to the
  sidewall corrugation, so it is LESS lossy (peak loss 0.04 vs 0.14) but has
  LOWER Q (770 vs 1500) and a WIDER mode (19 vs 14 µm). Its resonance sits at a
  different wavelength; TM needs its own pitch (516.8 vs 500 nm) to co-resonate.
- Matched peak transmission: TM needs 132 periods to match TE at 80, and there
  its Q is 2.9× TE's (4.9k vs 1.7k).
- Matched mode width (the program anchor): TM needs a deeper corrugation
  (400 vs 300 nm) to reach TE's 15.5 µm mode.
- **At the 20 µm / −3 dB point they are essentially equal: TM 13.9k vs TE
  12.9k (+8%).** Your "TM a bit higher, not considerably" is right.
- Where they differ is WHERE the light goes: TM radiates ~60% in-plane, with a
  sharp grazing "needle" 11–12° off the axis (ux ≈ 0.98); TE radiates upward in
  a broad cone with no needle. This is why every lateral structure below is
  TM-only (measured null for TE: trench, circle rows). Physics in one line: the
  TM mode has half the TE k-space margin to the cladding light line
  (n_eff − n_clad smaller), so envelope changes radiate more easily and the
  350 × 800 nm core is past the aspect ratio where the TM bandgap weakens.
MEASURED (job 97112, meeting 2026-06-22, `project_tm_radiation_design_rules`).

### 3.5 TM — apodization and tooth shifts
**Bottom line:** same qualitative behaviour as TE for apodization (T up, width
up a lot, e.g. 5 apodized periods = +15% width, 10 = +26%, 20 = +49%). Tanh
again widens less than linear. Tooth shift in TM: a large innermost shift raises
T (up to +0.03) but widens the mode (+5%) and shows no optimum; it does not
recycle the way TE does, because TM's loss lives in the arms, not at the corner.
**Correction:** per unit of added width the shift is actually slightly MORE
efficient than apodization; what apodization has is a higher ceiling. The "small
shift on all teeth" idea: a distributed shift is a loss-vs-width trade curve, not
a free win; fully spreading the π shift over many gaps is 20–40% WORSE than the
lumped π shift. Small shifts do earn their keep inside the width-constrained
inverse design (worth ≈ +0.025 T there, while apodization saturates), which is
where the "slight shifts do more" impression comes from. MEASURED (jobs 118293,
117530, 134977; `LOSS_EXPLORATION_FINDINGS.md`, `HANDOFF.md`).

**The general no-go (one equation, from the loss program):** the mode envelope
decays as exp(−∫q dx) with q = √(κ² − δ²) ≤ κ. Any tooth-level detuning δ
(apodization, shift, shape) can only LOWER q, i.e. lengthen the decay and widen
the mode. Narrowing needs κ raised near the centre, which only cavity-local
changes or new scatterers can do. Every measured width-reducing effect is in
that class and is ≤1% (air trench −0.9%, wider cavity −0.3%, circle row −0.4%).

### 3.6 TM — cavity-local levers (loss program, brief)
**Bottom line:** the cheapest loss lever is a wider cavity (800 → 1050 nm: loss
−30% at +0.6% width); adding an anti-symmetric "see-saw" to the two inner teeth
takes it to −31% at +0.8% width. Everything fancier at the cavity (curved shapes,
Hann profiles, exotic inner pairs, anti-radiator depth patterns, asymmetric
gratings) was measured null or worse; mirror symmetry is provably optimal.
MEASURED (jobs 117784, 117814; `LOSS_EXPLORATION_FINDINGS.md`).

### 3.7 TM — recycling the lateral radiation: air trenches
**Bottom line:** an air trench parallel to the waveguide, 1.8 µm away, is the
first lever that raises T AND leaves the mode width unchanged or slightly
narrower (−0.9%). At the 20 µm / −3 dB point: Q +35% (full-depth trench) or
+22% (trench stopping at the core's top surface, single-litho compatible, keeps
~60% of the gain). Mechanism: total internal reflection at the oxide→air wall
(the needle arrives at ~79°, critical angle ~44°) shrinks the in-plane light
cone. Decisive test: a metal wall at the same place gives an equal-magnitude
LOSS, so it is the low-index wall, not a mirror. A straight wall is exactly
optimal for a uniform grating (variational result); shaped/flared trenches were
measured worse. Trench gain stacks with apodization only partially and is a TE
null. Fab reality: a deep air etch conflicts with the PDMS acoustic cover,
which motivated the circles. MEASURED (jobs 124379, 126913; meetings 2026-07-21,
07-29, 08-05).

### 3.8 TM — the same effect with a row of SiN circles (latest result)
**Bottom line:** one row of SiN posts (r ≈ 80–110 nm, 350 nm tall, 1.8 µm from
the guide, period 531 nm) in the cladding does the trench's job in the same
lithography step: at the 20 µm / −3 dB point Q +16% (first lock, 57 posts r 80),
+26% (single 61-post row, r 110), +30% (a pair of 31-post rows centred at
±12 µm, r 110) — on par with the full trench. Mode width unchanged (−0.4%).

**Mechanism (the tiny bit of physics the advisor wanted):** the post row is a
weak grating for the guided carrier and out-couples a beam at the angle set by
n_clad·sin θ = n_eff − λ/Λ. Choosing Λ so that beam lands on the grazing needle
(Λ at the out-coupling cutoff, Λ_c = λ/(n_eff + n_clad) ≈ 531 nm) and then
SLIDING the row along x sets the interference phase φ = 360°·δx/Λ measured from
the cavity centre. The needle power follows a textbook sinusoid in φ: ×2.2 at
90°, ×0.45 at 270°. At 270° the out-coupled beam cancels the leak; at 0° (where
we started) it happens to be near the constructive side, which is why the first
period scans looked like "nothing helps, one period hurts". Once the phase was
scanned the gain appeared. An air (hole) version flips the sign, confirming the
interference picture. MEASURED (jobs 129989 → 146681; `COMB_HANDOFF.md`).

**Why a second row adds little:** the row's phase-independent cost is set by its
two ENDS (sign period ~12 µm along x), and its coherent gain by how much
carrier it sees. One row per side at ±12 µm is nearly additive (+0.019 vs
+0.012 for the best single, at N=80), but at the operating point that is only
+3% Q over one long row (18.1k vs 17.6k), and a third row adds ≤0.001 because
the carrier is too weak beyond |x| > 20 µm. Two rows stacked on the same
centre, more rows in y, radius apodization, full-length rows: all null or worse.
Circles do NOT stack with apodization (apodization already removes the needle);
they DO stack with the width-constrained inverse design. MEASURED (jobs
146364/146419/146564; `docs/comb_physics_rethink_2026-09-11.md`).

**Why the circles did not move in the inverse design:** their optimum (period
from n_eff, phase 270°, radius from amplitude matching) is separable from the
tooth envelope, so the optimizer sees no gradient on them; when freed they
drifted < 1 nm. Conclusion: design the row analytically outside the optimizer.
MEASURED (drift ≤ 0.66 nm x / ≤ 0.50 nm r).

### 3.9 Far-field / interference cancellation as a concept — what worked, what did not
- WORKED: the phase-controlled SiN circle row (above) — the program's first
  phase-controlled far-field cancellation. Air trench — works by light-cone
  shrinking, not interference.
- DID NOT: a 2-post SiN pair placed by a Green's-function response matrix
  (+0.02 T on the plain cavity but it is a near-field cavity-widening effect,
  vanishes on apodized/wide-cavity devices; dropped); period-matched retro-
  reflecting combs (transparent, ±0.001); metal/PEC mirrors (null to harmful);
  SiN DBR / photonic-crystal cladding walls (every variant loses: partial gap,
  reflected light never re-couples); in-core SiO₂ hole lattices (kill the
  resonance); forward-Huygens / Kerker scatterers and Friedrich–Wintgen BIC
  pairs (all raise loss); a second parallel cavity (drains, never recycles).
  All MEASURED (`LOSS_EXPLORATION_FINDINGS.md`, memory index).

### 3.10 Inverse design, current programme (lumopt2 adjoint, TM)
**What it is (say this, not the algorithm logic):** a 3D adjoint (gradient)
optimization in Lumerical's lumopt2 on a 100-period-per-side TM surrogate
(corrugation 325 nm). 191 parameters: for the 25 innermost teeth per side the
corrugation depth, average width and shift (mirrored), plus the cavity width,
plus the circle row (57 radii, 57 positions, standoff; frozen in practice). The
cost is peak transmission (a windowed soft-max of T(λ) around the resonance so
the passband maximum cannot fool it), with the spatial mode width held INSIDE a
±2% band around its seed value as a hard constraint, and the resonance wavelength
held fixed. The gradient step is projected so it changes neither the width nor
the resonance (both gradients come free from the same two adjoint solves). Q is
never in the cost: at fixed width Q_loaded is pinned, so every T gain is a
radiation-Q gain.

**What happened:** (1) a σ (second-moment) width proxy failed silently, the
true FWHM grew 15% while σ held; (2) a penalty-only cost failed twice (width
drifts to the band edge); (3) projection on width alone failed because the
resonance drifted and dragged the width; (4) projection on width AND resonance
works: seed T 0.90 → 0.964 (best, mostly a hand re-trim) → 0.968 machine-driven
with the resonance held to 0.0000 nm and the mode narrower, intrinsic Q 110k →
124k. At the 20 µm / −3 dB point the 0.964-class design measures Q 88,868.
The optimized profile is a gentle rise in corrugation over the 25 free teeth
(325 → ~364 nm), i.e. a mild version of the lab's overshoot.

**Status:** in progress, nothing running. Both lanes were stopped for optimizer
housekeeping bugs (not physics); fixes are written and gated locally, not yet
redeployed. Still below the lab's TE overshoot apodization. Ranked next: restart
→ free 60 teeth instead of 25 → free the circle row as a separate arm → a TE
lane seeded from the overshoot profile (TM intrinsic Q is capped around
150–250k by its light-cone margin; beating the TE overshoot needs a TE device).
MEASURED (`runners/lumopt2_design/HANDOFF.md`, `HANDOFF_2026-09-01.md`).

---

## 4. Status table (for the closing slide)

| direction | pol | status | verdict |
|---|---|---|---|
| Linear / tanh apodization | TE, TM | done | works; widens the mode; tanh widens less |
| Overshoot apodization (lab's model) | TE (TM checked) | measured, not ours | strongest known lever, TE ≫ TM |
| Innermost tooth shift | TE | done | loss down, width ~constant; ceiling small |
| Innermost tooth shift | TM | done | T up but widens; no optimum |
| 2-tooth inverse design (PSO / lumopt v1) | TE | closed | PSO T 0.97 at 16.5 µm; gradient not needed at 5 params |
| Cavity width + inner see-saw | TM | closed | −30% loss at <1% width; fab-simple |
| Air trench | TM | closed | Q +35% (full) / +22% (half) at operating point; TE null; fab conflict |
| SiN circle row(s) | TM | closed (delivered 2026-09-12) | Q +30% at operating point, same litho step; ≈ full trench |
| Reflectors, mirrors, hole lattices, BIC/Kerker, 2nd cavity, shaped trench, exotic teeth | TM (TE where tested) | closed | all null or harmful |
| 25-tooth adjoint inverse design, width-constrained | TM | in progress (paused) | Q 88.9k at operating point, 6.4× plain; below lab TE overshoot |
| Predictive engine (CMT-calibrated N/corr → T, Q, width) | both | validated | replaces tuning ladders with one confirmation run (not a device result; mention only if useful) |

---

## 5. Corrections and clarifications to the narrative you gave

1. TE tooth shift does not REDUCE the mode width; it keeps it nearly constant
   (15.4 → 15.9 µm) while cutting loss. Q rises modestly; it was never the
   headline metric in that phase.
2. TM at equal length has LOWER Q and LOWER loss than TE (both true); at the
   operating point the two are within 8% (TM 13.9k vs TE 12.9k), i.e. "a bit
   more, not considerably" is exactly right.
3. TM shift vs apodization: apodization widens MORE per unit gain, not less;
   its advantage is the higher ceiling. "Slight shifts on all teeth do more" is
   true only inside the width-constrained optimizer; a freely distributed π shift
   is worse than the lumped one.
4. Air trench: mode width unchanged to slightly NARROWER (−0.9%) — your
   "maybe even reduce it" is correct.
5. Circles vs trench: same outcome, DIFFERENT mechanism (interference of an
   out-coupled beam vs total internal reflection). The circle pair ties the full
   trench at the operating point (18.1k vs 18.8k).
6. "Zero phase, some period decreased transmission": yes — δx = 0 sits near the
   constructive phase, so the needle grew and T fell; scanning the phase found
   the 270° cancellation.
7. Second circle row: only a small gain at the operating point (+3%), for the
   end-rule / weak-carrier reason in §3.8. A third adds nothing.
8. Inverse design: 25 teeth PER SIDE (mirrored), 191 parameters including the
   circle row; 100 periods per side; the width band is ~18.3 µm in the
   optimizer's own mesher, which corresponds to ~20 µm in the project's standard
   mesher (the two meshers differ by 8% in width; never mix them).
9. "Better than the TE seed / overshoot": the inverse design (88.9k) is below the
   lab's TE overshoot projection (~2×10⁵–3.4×10⁵) but that projection was never
   measured at −3 dB; what IS measured is the intrinsic Q, which is 7–13× ours.
10. Chronology you gave is roughly right except: the pillar PAIR (Green's
    function) came before trenches (July), trenches came before circles (Aug),
    and the circle-pair result is the last thing done (2026-09-12).

## 6. Things I could not verify, or that you should decide

- TE overshoot at −3 dB: DECIDED (user, 2026-09-13) — quote the two measured
  routes, "~2.0×10⁵ (old scaled profile, Q_i 6.7×10⁵ at N=195) to ~3.4×10⁵
  (re-optimized Nt60 profile, Q_i 1.16×10⁶ at N=130), projected via
  Q(−3 dB) = 0.293·Q_i, still rising with N". Never the old chart's ~5.6×10⁵:
  it was a user-supplied extrapolation with no stored derivation (it would need
  Q_i ≈ 1.9×10⁶, ~120 periods beyond the last measured point).
- The inverse-design row without circles (~76k) is a scaled estimate, never
  measured. Either drop it or keep the "~".
- Two TE plain references exist (12,903 from the TE study; 12,741 re-measured at
  the inverse-design box). Use 12,903 everywhere except when quoting the "15×"
  Itai ratio, which was computed against 12,741.
- No 20 µm / −3 dB number exists for the d1 (0.968) inverse-design iterate; the
  88,868 is the 0.964-class design.
- The apodized-TM device at the operating point was never run (projected Q ~64k
  from a measured intrinsic Q of 217k at N=150, corr 400); do not put it in the
  table.
- Source reliability (user note 2026-09-13): later material supersedes earlier
  material. This briefing takes every NUMBER from the 2026-07 → 2026-09 result
  files and handoffs; the 2025-11 → 2026-04 decks contribute only direction
  words (apodization widens / shift keeps width), all of which later studies
  re-confirmed. The early CMT-cascading / overlap-correction thread (Dec 2025 →
  Feb 2026) is deliberately left out: it ended unresolved and nothing later
  depends on it. The TE tooth-shift numbers in §3.2 are the April 2026 ones
  (pre-HPC, N=80, pitch 500) and are quoted only as directions for that reason.
- Meeting decks quote several different TM baseline T values (0.80–0.89) for
  the "same" device because the simulation box changed; absolute T only
  compares within one study. That is why this briefing gives directions, not
  T values, outside §2.

## 7. Figures and data files to use in the deck (full paths)

Root: `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\`

- Benchmark bar chart, Q at the 20 µm / −3 dB point (2026-08-26; last bar still
  labelled ~5.6e5 and the circle-pair / full-trench rows are missing — re-render
  before use or caption accordingly):
  `results_from_athena\q20um_3db_benchmark\q20um_3db_benchmark.png` (+ `.fig`;
  script `matlab_plotting\plot_q20um_3db_benchmark.m`)
- Same devices at ~100 periods per side (T and Q at one common length):
  `results_from_athena\q20um_3db_benchmark\q20um_short_benchmark.png`
- Corrugation profiles, lab overshoot vs inverse design:
  `results_from_athena\q20um_3db_benchmark\tooth_profiles.png`
- Circle row: phase sinusoid and period scan:
  `results_from_athena\q20um_3db_benchmark\comb_phase_scan.png`,
  `results_from_athena\q20um_3db_benchmark\comb_period_scan.png`
- Cross sections of the short devices:
  `results_from_athena\q20um_3db_benchmark\short_device_cross_sections.png`
- Trench half/full cross-section slide:
  `results_from_athena\trench_flush_q3db\trench_normalized_cross_sections.png`
- Overshoot-apodization source data (the two Q_i routes in §2):
  `results_from_igum\itai_hh_summary.csv`, `results_from_igum\itai_hh_nt60w20_summary.csv`
- Per-device spectra/envelopes at the operating point (verified present):
  `results_from_athena\te_q3db_20um\te_q3db_20um_final_T_dB.png`, `..._final_envelope.png`, `..._T_Q.png`;
  `results_from_igum\trench_q3db_20um\trench_q3db_20um_final_T_dB.png`, `..._final_envelopes.png`, `..._T_Q.png`;
  `results_from_athena\invdesign_q3db_20um\invdesign_q3db_20um_final_T_dB.png`, `..._resonance_profile.png`, `..._T_Q.png`.
  The circle-pair device (Q 18,093) has NO plotted figure yet, only the .mat in
  `results_from_athena\comb_q3db_lock\results\`; the circle-row physics figures
  (leak location, off-centre pair, pair vs triple, summary) are
  `results_from_athena\comb_physics_rethink\fig1_leak_where.png` … `fig9_summary.png`.

## 8. Sources read this session
Meetings folder (all decks/docx, Nov 2025 → Aug 2026); memory files (all 123);
`runners/scatterers/COMB_HANDOFF.md`; `runners/lumopt2_design/HANDOFF.md`,
`HANDOFF_2026-09-01.md`, `THEORY.md`; `docs/comb_physics_rethink_2026-09-11.md`;
`results_from_athena/LOSS_EXPLORATION_FINDINGS.md`; per-study FINDINGS.md files;
`matlab_plotting/plot_q20um_3db_benchmark.m` and its figure;
`results_from_igum/itai_hh_nt60w20_summary.csv`. No file was modified or deleted.
