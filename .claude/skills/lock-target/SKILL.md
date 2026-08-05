---
name: lock-target
description: Tune a device to hit EXACT target values (spatial mode width, peak transmission / -3 dB point, resonance wavelength, Q via a loss knob) using the knob table + linearizing-coordinate ladder method. Use whenever the user asks for "exactly X µm mode", "peak T at -Y dB", "land the resonance at λ", "max Q given a T budget", or any find-parameter-for-target request — for any device or polarization.
---

# lock-target

Distilled from the trench_q3db_20um study (2026-08-02..04: 29 sims, two exact
targets locked in 2.5 rounds). The intelligence is the KNOB TABLE and the SOLVE
ORDER, not the root-finder — never reach for PSO/optimizers for target-hitting,
and do not build a generic framework (§10/§11).

## 1. Knob table (extend one row per new knob; keep entries measured)

| target | knob | linearizing coordinate | known side effects |
|---|---|---|---|
| spatial mode width `fwhm_m` | corrugation depth | 1/FWHM vs corr (FWHM = ln2/κ, κ ∝ corr) | changes T and Q strongly; retune N after |
| peak transmission (e.g. −3 dB) | `n_periods_each_side` | ln(T) vs N (locally linear) | width shifts only ~4% over ±30% N; λ unmoved |
| resonance λ | pitch | λ vs pitch (linear) | negligible on width/T; Δλ ≤ 1 nm acceptance |
| loaded Q at fixed T | NOT free: Q_L = (1−√T)·Q_i | — | needs a LOSS knob (trench, apod) as an extra dimension; those couple to everything — treat as a separate comparison arm, not a scalar target |

The system is nearly TRIANGULAR. Solve in table order: width → T → λ trim.
Multi-target requests are fine as long as each target has its own knob; if the
user asks for more targets than free knobs (e.g. width + T + Q with no loss
knob), say so — it is over-constrained, not a search problem.

## 2. Protocol (parallel ladder → fit → one confirm)

1. **Predict** the knob value from in-study data if any exists, else from the
   physics scaling. State the target λ and scan-window width in one line before
   dispatch (§4).
2. **Ladder**: one zipped `SweepSpec` with 3–5 points bracketing the prediction,
   dispatched as ONE array (max parallelism; hedged next-stage ladders may ride
   along — accepted rerun risk, call it out). Include the in-study no-change
   control row (§2).
3. **Fit** in the linearizing coordinate; solve for the target; check residuals
   (a good fit has |resid| ≪ tolerance — if not, the coordinate isn't linear
   here, add a bracket point instead of trusting the fit).
4. **Confirm** with ONE run at the solved (integer where applicable) value.
   In-band ladder points make the confirm free. If the confirm misses, it joins
   the fit and one more step is taken (regula falsi) — never redo the ladder.

Tolerance defaults (physical floors, don't tighten without reason): width
±1 µm (±0.25 µm on request); peak T ±0.03 — integer N quantizes T by 0.01–0.02
per period near T = 0.5, so tighter is impossible; Δλ ≤ 1 nm.

**Sibling-study shortcut (validated on te_q3db_20um, 2026-08-05):** the FIRST
study of a kind pays for full 4-5-point ladders to measure curvature; siblings
ride the measured line SHAPES with 2-point lines + a bracket/confirm pair
(TE ran ~8 sims vs TM's 29 for the same two targets). Two caveats, both
measured: (a) a T(N) line does NOT transfer across corrugation — dlnT/dcorr
was -0.05/nm at fixed N (corr 233→250 collapsed T 0.58→0.26), so re-anchor
T after every corr move; (b) when the crossing falls OUTSIDE the measured
pair, dispatch a bracket PAIR at the estimate (same wall-clock as one sim at
%2, converts extrapolation into interpolation).

## 3. Hard-won rules (each cost real GPU time)

- **Calibrate ONLY from in-study points at identical numerics.** Legacy anchors
  mislead: corr 300 = 19.1 µm in old data but 21.5 µm in-study → an 8-sim hedge
  ladder ran at the wrong corrugation.
- **Measure near the operating point.** The ideal cavity model is approximate —
  derived Q_i drifts with N (58k→76k over N 110→165). Don't extrapolate the
  decomposition far from where you'll operate.
- **Filename collisions**: at W800 the corrugation only enters the file tag via
  the TM `_C{corr}` branch in `sim_helpers.generate_file_tag` (added
  2026-08-02). Any NEW swept knob → verify tag uniqueness with a mock-sim
  smoke test BEFORE dispatch (§6 clobber).
- **Q is only reportable with ≥10 sample points across the spectral linewidth**;
  under-resolved points are excluded, not reported. Check per-point after every
  round: pts = |spectral_fwhm_nm| / (window/N_pts).
- **Serialize deploys** (§6): one study = one runner file, rounds are edits to
  its lists; redeploy only when the queue is empty. A transient scheduler outage
  can make `squeue` return empty — confirm "finished" with two consecutive clean
  polls or `scontrol show job`, never a single empty read.
- Sequential secant drivers (`runners/tm/tm_match_pitch_bisect.py`,
  `tm_wide_mode_corr.py`, …) remain the right tool when only one GPU/seat is
  free or the search is 1-D and cheap — but note the deploy `--export` list may
  not forward all their env knobs, and their caches are corr-keyed only (not
  N/pitch-aware) — check both before reusing.

## 4. Speed levers (apply by default; measured on trench_q3db_20um)

Wall-clock split there: ~60% license-throttle queueing, ~25% solve, ~15% round
boundaries. Attack in that order:

1. **Fill all 6 global license seats — by splitting STUDIES, not comparisons.**
   Run the next study/round on the idle cluster while the current one drains
   (separate sweep_lists = no §6 clobber; node diversity also dodges same-node
   license-daemon races). NEVER split anything numerically compared across
   clusters: convergence curves, sweep-vs-control deltas, and the final
   head-to-head confirms of a comparison all share ONE cluster (offset
   ΔT ~0.004 / Δλ ~2 nm ≈ the effects being chased). Ladders that are only
   fitted internally may live on either cluster, whole. Sum of throttles ≤ 6;
   never launch into a full house — and note squeue-empty ≠ seats-free (other
   users share the pool; the license race dies instantly, so casualties are
   cheap: resubmit dead indices staggered via --array-tasks). The 6-seat ceiling
   is an UPPER bound (faculty-shared pool; 4+2 across clusters died on it
   2026-08-04). Starvation signatures differ: IGUM native = loud "Unable to
   checkout"; Athena container = SILENT no-op ("Simulation time: ~1 s", then
   "Can not find result 'expansion for port monitor'" — check solve time before
   blaming the .h5-clobber cause). Opening a second cluster or resuming after
   any license anomaly: ONE canary task first, fleet only after it shows a real
   solve time.
2. **Exclude known-slow nodes.** ece-ykasten1 ran identical solves 1.5–3×
   slower than efrats nodes (measured twice). `--exclude=ece-ykasten1` on the
   sbatch / in igum.conf.
3. **Pipeline round boundaries.** Fit after every fetched batch; as soon as the
   prediction stops moving (typically 3 of 5 ladder points in), dispatch the
   next stage — keep the license seats continuously full. Hedge dispatches are
   the same idea; state the rerun risk when taking it.
4. **Short-device κ calibration for the width knob.** κ from the stopband width
   of an N≈40 device (minutes/sim, broad features) places the corr prediction;
   only 1–2 full-size sims confirm. Model-mediated (FWHM = ln2/κ) → the
   full-size confirm is MANDATORY (in-study-anchors rule).
5. **Auto-shutoff threshold: SETTLED 2026-08-04 (study autoshutoff_qspan) —
   1e-7 for EVERYTHING, do not relitigate.** Truncation error is a function of
   Q ONLY (five devices, TE+TM, plain/trench/apod collapse on one ~Q^0.7
   curve); 1e-6 costs 2% Q already at Q~1.5k and 16% at Q~27k. 1e-8 is
   UNREACHABLE (total-field energy floor ~5e-8 -> rows run to the time cap and
   die on wall-time). 1e-7 = floor + one spare decade. Knob:
   `cfg.mesh.auto_shutoff_min` (None = 1e-7). The only open edge: Q >~ 1e5
   devices (h200 family) are extrapolation — first resumed ladder there
   carries one strict-vs-relaxed guard pair. Keep-forever data:
   results_from_igum/autoshutoff_qspan/.

## 5. Reporting

Per locked target: knob value, measured observable, residual vs target, and the
§2 sanity gates. Label numbers MEASURED / DERIVED / EXPECTED. Worked example +
final numbers: `results_from_igum/trench_q3db_20um/` and
`runners/metal_mirror/trench_q3db_20um.py`.
