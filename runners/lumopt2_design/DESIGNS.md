# lumopt2 corr-325 program — design registry (updated 2026-08-17 ~17:10)

> ★★★2026-08-18 — **EVERY `sigma` AND `FWHM` COLUMN BELOW IS VOID.** `profile_line`
> never integrated over y (it always returned y-row 0), so all widths recorded
> here were measured on one off-axis row. T / lambda / Q_L / Q_i / R / loss are
> port quantities and remain valid. Corrected widths and the current program
> state are in **`HANDOFF.md`** (same directory) — read it before using this file.

Every named design, its measured metrics, and where its FULL 191-param vector
lives **locally** (no server dependency). Param layout: 25 corr | 25 avg |
25 shift | 57 r | 57 x | d_comb | cavity_w (nm). sigma0 (width ref) = 17.493 um.
Local logs: `results_from_athena/lumopt2_c325_logs/` + `results_from_igum/
campaign_c325_seedB/`. All metrics MEASURED at campaign numerics (N=100
surrogate, y6.8/z6.8, PVA mesh, 301 pts @ 20 pm) unless noted.

| design | T | sigma (um) | ratio | Q_i | FOM | full vector (local) |
|---|---|---|---|---|---|---|
| uniform seed (stage-1 A start) | 0.8924 | 17.489 | 0.9998 | 36,868 | 0.65934 | seedA jsonl, eval 1 |
| dip seed (gen-1 B start) | 0.9167 | 17.444 | 0.9972 | 48,743 | — | seedB jsonl, first eval-1 row |
| seedA stage-1 best (eval 8) | 0.9313 | 17.7519 | 1.0148 | 57,936 | 0.68831 | `campaign_c325_seedA2.py::SEED` + seedA jsonl |
| **seedB best (eval 17)** | **0.9460** | 17.7516 | 1.0148 | ~69k | **0.70045** | seedB jsonl (fetched 16:35) |
| seedB eval-5 (A/B-verified) | 0.9451 | 17.705 | 1.0121 | 73,722 | 0.70011 | `comb_dip_ab.py::P_BEST` |
| stage-2 (133499 ev2, pre-crash) | 0.9375 | 17.7506 | 1.0147 | 63,937 | 0.69291 | seedA2 jsonl, row 2 |
| stage-2 (133530 ev2) | 0.9407 | 17.7521 | 1.0148 | 67,567 | 0.69586 | seedA2 jsonl, row 4 |
| stage-2 (133530 ev3) | 0.9609 | 17.7914 | 1.0171 | 103,149 | 0.71213 | `best_designs.py::BEST_T9609` |
| **★PROGRAM BEST (133530 ev4)** | **0.9635** | 17.7952 | 1.0173 | **110,874** | **0.71409** | **`best_designs.py::BEST_T9635`** + seedA2 jsonl row 6 |
| stage-2 final (ev5, NOT the seed) | 0.9636 | 17.8186 | 1.0186 | ~111k | 0.71420 | seedA2 jsonl row 7 (+0.0001 sub-jitter for +0.024 um width -> rejected as seed) |
| stage-3 baseline (= seed, H200) | 0.9318 | 17.7490 | 1.0146 | 58,390 | 0.68971 | seedA3 jsonl, row 1 |
| bare seed (uniform, no comb) | 0.8807 | — | — | — | — | bare jsonl, eval 1 |
| bare after 1 step (55343 ev2) | 0.9212 | 17.610 | 1.0067 | — | 0.68210 | bare jsonl, eval 2 |
| tangent: shift-only (+40 nm) | 0.9409 | 17.8985 | 1.0232 | 66,622 | probe | SEED with shifts ×1.3063 |
| tangent: corr-only (+5 nm) | 0.9298 | 17.705 | 1.0121 | 56,974 | probe | SEED with corr +5.0 |
| tangent: combo (+80/+7.54) | 0.9440 | 18.0053 | 1.0293 | 69,462 | probe | SEED ×1.6126 / +7.54 |
| A/B: eval-5 with comb (Athena) | 0.94629 | 17.7045 | 1.0121 | 75,361 | 0.70066 | = P_BEST |
| A/B: eval-5 comb REMOVED | 0.94147 | 17.71196 | 1.0125 | 68,900 | 0.69743 | P_BEST, comb inert |

Comb basin scan (job 133718, base = seedB eval-5; anchors comb 0.94629 /
no-comb 0.94147 at identical numerics):

| variant | T | Q_i | sigma (um) | lambda | vs comb | vs no-comb |
|---|---|---|---|---|---|---|
| phase +90 deg (task 0) | 0.93958 | 66,626 | 17.7119 | 1565.914 | -0.0067 | -0.0019 |
| phase +180 deg (task 1) | 0.93333 | 60,078 | 17.7173 | 1565.914 | **-0.0130** | **-0.0081** |
| phase +270 deg (task 2) | 0.94006 | 67,180 | 17.7098 | 1565.914 | -0.0062 | -0.0014 |
| pitch 516.83 = grating (task 3) | 0.94144 | 68,861 | 17.7117 | 1565.914 | -0.0049 | **-0.00003** |
| pitch 524.0 (task 4) | 0.94374 | 71,843 | 17.7048 | 1565.914 | -0.0026 | +0.0023 |
| pitch 540.0 (task 5) | 0.94165 | 69,111 | 17.7089 | 1565.914 | -0.0046 | +0.0002 |
| radius 70 (task 6) | 0.94591 | 74,819 | 17.7054 | 1565.914 | -0.0004 | +0.0044 |
| radius 100 (task 7) | 0.94547 | 74,124 | 17.7044 | 1565.914 | -0.0008 | +0.0040 |

★FAB TOLERANCE SUMMARY (all four axes now measured on the same device):
  phase  : SHARP  — a quarter period costs 0.0067; half a period costs 0.0130
           and is worse than having no comb at all.
  pitch  : TIGHT  — hold to ~+/-3 nm; -7 nm keeps half the benefit, +9 nm none.
  radius : LOOSE  — 70/80/100 nm span only 0.0008 = BELOW the jitter floor.
  count  : ★n=29 (HALF the comb) = T 0.96104 vs n=57 control 0.9609 -> +0.0001,
           a DEAD TIE 20x below the jitter floor. The outer 28 posts do
           nothing. PREDICTION (recorded before the run) CONFIRMED, and via
           the mechanism: k-space length matching needs L_comb ~ L_mode, and
           57 posts = 29.7 um was already LONGER than the ~17-21 um matched
           band; 29 posts = 14.9 um still covers the needle's angular width.
           => FAB SIMPLIFICATION AVAILABLE FOR FREE: halve the post count.
           True optimum likely ~40-45 posts, but the difference from either
           measured point is sub-floor, so not worth chasing.
           ★n=113 MEASURED T 0.96167 (+0.0008) — MY PREDICTION ("clearly
           worse") IS FALSIFIED. Full series at identical numerics:
             n=29  (+/-7.4 um)  T 0.96104   +0.0001 vs ctrl
             n=57  (+/-14.9 um) T 0.9609    control
             n=113 (+/-29.7 um) T 0.96167   +0.0008
           ALL within the ~0.002 jitter floor => the post count DOES NOT
           MATTER over 29..113, a 4x range in comb length, in either
           direction. The k-space matching model predicted a knee at BOTH
           ends; the UPPER end is now falsified.
           ★DO NOT invent a mechanism for the upper-end flatness: the obvious
           "outer posts sit in the dark" story FAILS arithmetic — with sigma
           17.8 um the n=113 edge posts sit where intensity is still ~9% of
           peak (and n=29 stops at ~56%), so they ARE illuminated and still
           change nothing. Mechanism UNSETTLED; logged as such.
           => the interesting knee is BELOW 29. Close-out sweep revised to
           n in {7, 13, 21}; DROP the planned 41 (confirmed-flat region).
           => RECOMMENDATION STANDS: n=29, same performance at half the posts.

★Reading so far: rotating the comb is MONOTONICALLY harmful in both T and
Q_i (75,361 -> 66,626 -> 60,078), and past 90 deg it is worse than having NO
comb at all (-0.0081 at 180 deg). So the comb is strongly phase-sensitive
(0.0130 swing = 6x the jitter floor) AND the optimizer leaves it within
0.7 nm — consistent only with the comb sitting at a SHARP local maximum.
Both rotations are lambda-identical (1565.914) and width-neutral (sigma
17.71), so this is pure radiation loss, no resonance/width side-channel.
Reproduces the air-comb pi-flip on the apodized+shifted device.
★★COMMENSURATE PITCH = COMB SWITCHED OFF (task 3): pitch 516.83 (= grating)
gives T 0.94144 vs 0.94147 with NO comb at all — 3e-5 apart, 50x below the
jitter floor. The comb becomes exactly neutral.
★CORRECTION (user, 2026-08-17 night — an earlier note here claimed a
"beat with the grating" mechanism; that was a post-hoc invention, RETRACTED).
The pitch was DERIVED from the RADIATION LOBE ANGLE via the grating equation
validated in the anti-needle study (stage-O fit to 0.001):
      n_eff = lambda/Lam_comb - n_clad*|u_x|
i.e. Lam_comb is chosen so the comb's first-order out-coupled beam lands ON
the grazing needle and cancels it (engineered Friedrich-Wintgen quasi-BIC).
Commensurate pitch aims that beam at the grating's own Bragg order instead of
the needle, so the cancelling function disappears — consistent with the
measurement, and the correct reason for it.
★DERIVED for the CURRENT device (does apodization+shift move the aim?):
n_eff = lam_res/(2*Lam_grating) = 1566.16/(2*516.83) = 1.5151; at Lam_comb 531
=> |u_x| = (lam/Lam - n_eff)/n_clad = 0.993, still on the measured ~0.99
needle. Required pitch = lam/(n_eff + n_clad*|u_x|): 530.7 nm at design time
(lam 1559) -> 531.9 nm now (lam 1566.16) = a 0.9 nm shift ONLY, because
lam_res and n_eff move together at fixed grating pitch. Sensitivity from the
516.83 point: dT/dLam ~ 3.4e-4 /nm => 0.9 nm is worth 0.0003 T, an order of
magnitude BELOW the jitter floor. That is also why the gradient leaves the
comb alone: a 0.9 nm pitch fix = ~25 nm on the outer posts, well inside their
+/-100 nm bounds, so it COULD move and correctly does not.
★★THE WHOLE PITCH SERIES IS EXPLAINED BY THE LIGHT LINE (2026-08-17, after
task 4). With n_eff = lam/(2*Lam_grating) = 1.51492 (this device's own Bragg
condition at the scan base lam 1565.914) and |u_x| = (lam/Lam - n_eff)/n_clad:
      pitch 516.83 -> |u_x| 1.0491  EVANESCENT, order cannot propagate
      pitch 524.00 -> |u_x| 1.0204  EVANESCENT
      pitch 531.00 -> |u_x| 0.9931  radiating, ON the ~0.99 needle  <- in use
      pitch 540.00 -> |u_x| 0.9591  radiating, ~3 deg off the needle
LIGHT-LINE CUTOFF = 529.22 nm. Below it the comb's diffracted order does not
exist as a propagating wave, so the comb has nothing to interfere with — THAT
is why 516.83 reproduces the no-comb value exactly (supersedes the earlier
"commensurate" framing: being below cutoff is the operative fact, and the
commensurate pitch merely happens to sit in that dead zone). Note how tight
the design point is: 531 is only 1.8 nm above cutoff.
★PREDICTION RECORDED BEFORE THE MEASUREMENT (task 5, pitch 540): should fall
BELOW 531 (expect ~0.943-0.945) because it aims off the needle. If instead it
comes out ABOVE 0.9463, the needle itself has MOVED on the apodized+shifted
device and the far-field readout becomes the priority.
★RESULT: 0.94165 — BELOW 531 as predicted (direction right; magnitude a bit
low vs the 0.943-0.945 guess, so the off-needle fall-off is steeper than the
linear estimate). PITCH CURVE NOW BRACKETED BOTH SIDES, PEAK AT 531:
  516.83 -> +0.00000 vs no-comb | 524 -> +0.0023 | 531 -> +0.0048 | 540 -> +0.0002
★★THEREFORE THE NEEDLE HAS NOT MOVED on the apodized+shifted device — had the
lobe angle shifted, the optimum pitch would have shifted off 531 and it did
not. This settles the user's lobe-angle question EMPIRICALLY (the far-field
readout would still be the direct confirmation, now a nicety not a priority).
★FAB TOLERANCE (derived from the four points): the benefit collapses either
side of 531 — about half retained at -7 nm, essentially none at +9 nm — so
hold the comb pitch to ~+/-3 nm. Asymmetric because going SHORT crosses the
529.22 nm light line (comb switches off) while going LONG stays radiating but
aims progressively wide of the needle.
★★WHY THE COMB IS INVARIANT — the user's "it only depends on the mode width"
reading, checked against the record and CORRECT with one refinement. The comb
has TWO properties with DIFFERENT dependences:
  (a) WHERE IT AIMS (pitch, phase) = grating equation, |u_x| = (lam/Lam -
      n_eff)/n_clad. Depends on lambda and n_eff. NOT on the mode width.
      -> that is why the pitch optimum stayed at 531 after the grating was
      heavily reshaped (measured tonight, tasks 3-5).
  (b) HOW WELL IT CANCELS (length = count x pitch) = k-SPACE OVERLAP. The
      comb's beam has angular width ~1/L_comb; the needle has angular width
      ~1/L_mode; they must MATCH. Depends on the mode width - exactly the
      user's intuition. Stage-P recorded it explicitly: "31 posts = +/-8.2 um
      ~ 20% of device - ANGULAR width matching, not coverage; cancellation is
      k-space not real-space", and "needle width 0.04-0.05 needs L ~ 17-21 um;
      N41 ~ tie, N61 BELOW".
=> Holding the mode width FIXED freezes both -> the comb cannot want to move.
   The "mirror that reflects the mode width" picture is right in spirit; the
   matching is angular, not real-space coverage.
★PREDICTION RECORDED BEFORE THE COUNT MEASUREMENT (job 133793): our comb is
57 posts = 29.7 um long, ALREADY LONGER than stage-P's matched 17-21 um band
(scaled for our larger mode, maybe ~20-25 um). So expect n=113 (59.5 um)
CLEARLY WORSE, and n=29 (14.9 um) TIE-or-slightly-better vs n=57 - which
would put the true optimum near 40-45 posts, i.e. our 57 is a bit long.
OPEN (needs 1 sim): far-field readout of BEST_T9609 at resonance to MEASURE
the needle angle on the apodized+shifted device instead of assuming 0.99.

Shift ladder (job 134033, on BEST_T9635; control x1.0 = the winner itself):

| scale | 2*Sig_s | T | Q_i | sigma (um) | ratio |
|---|---|---|---|---|---|
| x0.0 (shifts deleted) | 0.0 nm | 0.93613 | 63,994 | **17.4956** | **1.0001** |
| x0.5 | 65.3 nm | 0.95222 | 85,932 | 17.5884 | 1.0055 |
| x1.0 (control, stored) | 130.6 nm | 0.9635 | 110,874 | 17.7952 | 1.0173 |
| x1.5 | 195.9 nm | 0.96747 | 120,550 | 18.0620 | **1.0325 OUT** |

★★THE SHIFTS SIT AT THE CONSTRAINT BOUNDARY, NOT A PHYSICS OPTIMUM:
  T rises monotonically (0.9361 -> 0.9522 -> 0.9635) but sigma rises
  SUPERLINEARLY (+0.093 then +0.207 um for equal shift increments), so the
  efficiency COLLAPSES 3x across the two rungs: 0.173 -> 0.055 T per um.
  Quadratic fit predicts x1.5: sigma 18.116 um (ratio 1.0356) = OUT of the
  +2% band by 0.27 um, for only ~+0.0065 T.
  ★PREDICTION RECORDED BEFORE THE MEASUREMENT: x1.5 comes back NON-COMPLIANT.
  => stage-1 did not "choose" 130.6 nm; it pushed the shifts until the WIDTH
  WALL stopped them. The remaining room is NOT in bigger shifts (width-
  blocked) but in the sigma-NEUTRAL trades stage-4 is searching (bigger
  shifts paid for by corrugation elsewhere).

★★TWO BIG READINGS (2026-08-18):
1. The shifts are the program's MOST VALUABLE feature: deleting them costs
   **-0.0274 T** (14x the jitter floor) and **-42% Q_i** — 5.7x the whole
   comb's +0.0048. "Do the shifts still earn their place?" -> emphatically yes.
2. ★The ENTIRE width excursion is theirs: with shifts removed sigma returns to
   17.4956 = ratio **1.0001**, i.e. essentially sigma0. So corrugation and
   cavity width (which produced most of stage-2's gain) are NET WIDTH-NEUTRAL,
   and the whole +1.7% of band in use is bought by ONE mechanism — which is
   also the most valuable one. Efficiency averaged over 0->130.6 nm is
   0.091 T/um vs the LOCAL slope 0.065 at x1.0 => already in diminishing
   returns (concave), matching the "sigma superlinear / T sublinear" finding.
CAVEAT: x0.0 is the winner with shifts DELETED, not a re-optimized shift-free
design — this measures their contribution at this operating point only.

Key measured levers (for figure-making and continuation):
- shift (+2Sigma_s): dT +2.4e-4/nm, dsigma +0.0037 um/nm (superlinear ~+16%/40nm)
- corr (free 25): dT −0.123/rho, dsigma −3.85 um/rho, lambda-NEUTRAL
- cavity y-width: strong T lever, sigma-flat, lambda-neutral (+13.4 nm in one step)
- comb removal: −0.0048 T on the dip design (mechanism ~83% preserved vs origin)
- **comb re-tuning: MEASURED NULL.** Across 3 consecutive accepted optimizer steps
  (stage-2 rows 1-4) the comb moved r_mean +0.0065 nm, x_rms 0.024 nm, d_comb
  -0.4 nm — i.e. motionless to ~30 pm while cavity width moved +25.8 nm and the
  inner corrugations moved -10 nm in the same steps. The comb is at a local optimum
  of its own geometry; it EARNS its +0.0048 T by being present, not by being tuned.
- **stage-2 mechanism (what actually moves at frozen shifts):** cavity y-width
  812.7 -> 826.1 -> 838.5 nm (+12.4 nm/step, monotone, sigma-flat) and the inner
  corrugation dip deepening (corr_1 316.8 -> 311.4 -> ~306 nm, rho 0.9968 ->
  0.9938). Both were nearly frozen in stage-1 -> stage-1 was gradient-starved on
  them, not converged.
- **repeat noise:** the same parameter vector re-measured on a different node gave
  T 0.9375 vs 0.9357 (dFOM 0.0006). Treat ~0.002 in T as the per-eval jitter floor
  (matches the CLAUDE.md section-2 dx=50 nm floor); single steps below that are not
  results, the 0.9318 -> 0.9407 trajectory is.

Live (2026-08-18 ~01:00): **stage-4 = Athena 134032** (seed BEST_T9635, all
191 free, sigma-hat wall + trust_nm — THE continuation); **shift ladder =
Athena 134033** (x0 / x0.5 / x1.5 on BEST_T9635, control = stored winner);
**count study = Athena 133793** (n=29 / n=113 on BEST_T9609, control n=57);
**bare = IGUM 55801** (trust_nm resume from its log, ev5 T 0.9249).
CLOSED: stage-2 133530 (winner banked), stage-3 133541 (obsolete seed),
bare 55343 (lnsrch death -> engine fixes), comb basin scan 133718 (9/9,
comb optimal in every scanned direction).
Registry maintenance rule: fetch the small jsonl logs to the local dirs on
EVERY milestone check (CLAUDE.md §6 fetch-early rule) and refresh this table.

## ★★★THE FWHM PROBLEM (opened 2026-08-18, job 134217) — READ BEFORE TRUSTING ANY WIDTH NUMBER

The campaign controlled **sigma** (2nd moment) for its whole duration. The
ACOUSTIC SPEC is **spatial FWHM**. FWHM was first logged 2026-08-18. Measured:

| design | T | sigma | sigma ratio | FWHM | FWHM/sigma |
|---|---|---|---|---|---|
| uniform ORIGIN (134217 t0) | 0.8926 | 17.487 | 1.000 | **17.100** | 0.978 |
| d+20 = new best (134107 t0) | 0.9659 | 17.818 | 1.019 | **22.210** | 1.247 |
| d+40 | 0.9667 | 17.851 | 1.021 | 22.224 | 1.245 |
| d+60 | 0.9663 | 17.891 | 1.023 | 23.208 | 1.297 |
| d+80 | 0.9653 | 17.938 | 1.025 | 23.243 | 1.296 |

★sigma grew **+1.9%** while FWHM grew **+29.9%**. The +2% sigma band therefore
did NOT enforce the spec: a 2nd moment is blind to a FLATTENING CORE, and the
optimizer flattened the core while leaving the tails such that sigma barely
moved. FWHM/sigma 0.978 -> 1.247 IS that shape change, measured.
CONSEQUENCE: every "in-band" claim in this file means IN THE SIGMA BAND. It
does NOT mean the design meets a 20 um FWHM target. The winner BEST_T9635 is
pending its own FWHM row (134217 t1), and the shifts-zeroed control (t2) will
say whether the shifts CAUSED the broadening (recoverable with the same lever)
or the corr/cavity shaping did (sigma never had authority over it).
GUARDS SHIPPED same day: every eval now logs mode_fwhm_um + fwhm_over_sigma and
ALARMS when the ratio drifts >0.05 from the origin's 0.978; the width surrogate
also logs predicted-vs-measured with its own alarm (skill items 24-25).
