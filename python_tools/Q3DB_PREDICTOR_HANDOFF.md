# Q3dB PREDICTIVE ENGINE — SELF-CONTAINED HANDOFF (2026-09-01)

Hand this file to a session with no memory of the work. It is the complete state:
what the program is, what it can do, every measured validation, how to run it,
what is parked. Companion memory file:
`project_q3db_predictive_engine.md`; skill: `predict-q3db`.

---

## 1. What this program replaces

The Q3dB workflow used to be: pick a device, then run a LADDER of long FDTD
simulations, tuning corrugation (for mode width) and N (for peak T) until the
device sits at −3 dB with the right mode width — then quote Q. That costs many
long runs, and above Q~1e5 the devices cannot be simulated reliably at all
(the 5 pm spectral grid and the 2000 ps ring-down both break, and both bias T
low in a self-confirming way).

Now: **predict → ONE confirmation run → refit.** The engine is calibrated from
results already on disk; it was hold-out backtested 46 ways and then validated
on three devices dispatched specifically to test it.

## 2. The model (three layers)

**L0 — exact two-port algebra** (validated ~2-3% on four measured anchors):
```
1/Q_L = 1/Q_i + 1/Q_c        sqrt(T) = Q_L/Q_c        Q_i = Q_L/(1-sqrt(T))
T = (Q_i/(Q_i+Q_c))^2        Q(-3dB) = 0.29289*Q_i  [only if Q_i saturated]
conditioning A = sqrt(T)/(2(1-sqrt(T)))   -> never fit Q_i from rows with A>~5
```

**L1 — coherent lane (per family, extrapolates well):**
- `Q_c(N) = exp(lnQ0 + rate*N)` — rate = 2*kappa*Lambda. THE extrapolation lane.
  **Never extrapolate ln T** (measured: that missed a crossing by +191%).
- width `F(N) = F_inf - B*exp(-c*N)` (held-out error 0.4-0.7%).
- lambda from the pitch/corr line; N-independent to ~0.1 nm.
- `bragg_cmt.py` (piecewise Erdogan CMT/TMM) is the FIXED-N SHAPE tool:
  apodized-width prediction (<1%), full spectra, and Q_c as a shape RATIO
  anchored on one measured row. Its N-trends are NOT primary (engine width-vs-N
  is flatter, crossover Q_c steeper than measured).

**L2 — radiation lane (per family, the risky half):**
- `Q_i(N) = A*N^p` with **saturation**: `1/Q_i = 1/(A*N^p) + 1/Q_sat`.
  A pure power law fitted through the knee produces garbage exponents.
- Q_i is the failure surface of every historical attempt (it drifts +31% below
  saturation; saturation criteria do NOT transfer between devices).
- corrugation moves Q_i as ~corr^−2.9 (MEASURED, TM, fixed N=150).

**Polarization** is structural: families are single-pol by name (`tm_*`, `te_*`,
`itai_*`), fitted only on their own rows; the width↔corr knob lines are per-pol.
**Mesher frames** never mix: the calibration is all conformal (q3db family
numerics); a PVA-frame row triggers a loud warning (offsets: λ +5.3 nm,
FWHM −8%, T +0.008, Q_L −7%).

## 3. The tools (`python_tools/`)

| file | what it is | how to run |
|---|---|---|
| `bragg_cmt.py` | CMT/TMM engine: kappa(z) apodization, pi/fractional plates, z-dependent complex loss, envelopes, spectra | `python python_tools/bragg_cmt.py` runs its gate suite (incl. deliberate-failure checks). Do this after ANY edit. |
| `calibrate_q3db.py` | fits every family from STORED results, runs backtests B1-B14, writes the calibration CSV | `python python_tools/calibrate_q3db.py` from repo root. THIS IS THE VERIFICATION — rerun after any new result or model change. |
| `predict_q3db.py` | the design/prediction tool | edit the knobs at the top, run it |
| `q3db_calibration.csv` | fitted parameters + provenance (engine version, numerics) | data, read by predict_q3db |

`predict_q3db.py` modes: **observe** (family + N → observables), **design**
(target dB + optional width target → corr*, N*, expected observables + the
confirmation-run spec), **extend** (anchor on ONE new measured row, borrow the
family shape — the "here is my new result, extend it" workflow).

## 4. Backtests: 44/46 gated pass

The 2 failures are DELIBERATE stress rows (B2-E), kept failing to mark the
validity boundary. Headlines, all out-of-sample:

| test | what | result |
|---|---|---|
| B1b | fit invdesign N=100-200 → held-out N=220 | Q_L +2.3%, T −1.4pt, crossing −0.3% |
| B4b-live | fit stored c276 N=110-165 → **dispatched** N=200 | Q_L +1.9%, T −1.0% |
| B5 | width fit N=60/70/80 → held-out N=100/120 | 0.3-0.5% |
| B7 | Itai TE Q_i at N=175/195 (>1e5 regime) | 4.7-5.6% |
| B10 | fit κ,n_eff on ONE short spectrum → N=165 lineshape | λ 0.01 nm, Q_c +6.7% |
| B11 | apodized width A2→A20 via kappa(z) CMT | all <1% |
| B13-TE | TE-only fit N=166-190 → held-out N=215 | Q_L +2.2%, T +1.7% |
| B14 | κ∝corr at corr 448 (+38% outside range) | rate 1.0%; 2-term Qc transform −7.6% |

**Walk-forward on the q3db ladder** (fit only what was measured at each stage):
next-rung T within 1.7 pt and Q_L within 3.5%; Q_L at the −3 dB device came in
at +14.2% (2 rungs) → +6.7% (3) → +2.3% (4).

## 5. The three live validation runs (2026-09-01, IGUM)

Predictions were PRE-REGISTERED in each runner's docstring before dispatch.

**67731 — `runners/sweeps/tm_nladder_c276.py`, c276 N=200.** Tests whether the
fitted Q_i saturation is real, 35 periods beyond the family's data.
MEASURED T 0.5696 (pred 0.5641), Q_L 19234 (pred 19599, −1.9%), width 23.91
(−0.1%), λ 1559.92 (−0.01 nm). **PASS, both bands.**

**68086 — `runners/sweeps/tm_q3db_14um_knob.py` rung 0, corr 448.4 / N=98.**
One-shot −3 dB at a 14 µm mode — a corrugation the calibration had never seen.
MEASURED T 0.5808 (ABOVE band), Q_L 3853 (−16.9%), **width 14.15 µm (+2.1% —
the width knob worked)**, λ 1557.75. Decomposition: **Q_i +3.8% (corr^−2.9
validated); the whole error was in Q_c** — the corr transform carried the rate
(κ∝corr) but not the level. Zero-GPU fix: two-term transform, intercept
−0.002818/nm fitted on the STORED N=150 corr ladder (residuals ≤3.5%),
post-hoc −7.7% on this row. Now `QC_H_PER_NM` in `predict_q3db.py`.

**68925 — same study, rung 1, N=103** (re-designed with the fix + rung 0 as
anchor). MEASURED T 0.5097 (pred 0.512), Q_L 4644 (pred 4666, −0.5%), width
14.19 µm (+0.6%), λ 1557.75 (+0.05 nm). **PASS on all four.**
⇒ **a −3 dB / 14 µm device delivered in TWO runs at an unseen corrugation.**

## 6. Rules that carry (each one was paid for)

- Extrapolate Q_c, never ln T.
- Q_i needs the saturating form; distrust Q_i near/below onset.
- κ is linear in corr (0.1-1.3% over 276-400, 1.0% at 448); Q_i(corr) is a
  SEPARATE radiative law (~corr^−2.9). These are different channels — the
  three conflicting exponents in the old notes were this confusion.
- **A knob transform needs BOTH terms** (rate AND level). Validate any knob
  against the stored ladder in that knob at fixed N BEFORE dispatching.
- **Decompose every miss into Q_c and Q_i before touching the model.**
- Never calibrate κ on a Q level (ill-conditioned); use widths or Q_c growth.
- Anchor/calibration devices need `2*kappa*L >= ~3.2` (c325: N ≥ ~93). The
  tool refuses shorter rows.
- T ±0.03 holds ~30 periods beyond the anchored range; band by ~45.
- One anchored row + one confirm run makes a NEW family design-grade.
- Pre-register predictions in the runner docstring before every confirm run.

## 7. Why the earlier CMT attempts failed

1. **Constant-α CMT cannot carry this radiation.** It predicts Q_i independent
   of length; measured Q_i grows ~N^3 then saturates. Such a fit HAD to
   half-work. Radiation must be its own calibrated layer.
2. **Bugs in the MATLAB CMT project** (user's separate repo, NOT fixed by us):
   a dz-sign convention that turns loss into GAIN in the one driver exposing
   loss; an S↔T convention mismatch corrupting the reflection-phase correction;
   T>1 at the CMT↔FDTD seam patched by a one-sided `junctionEta` fudge; four
   inconsistent spatial-FWHM definitions and ~6 Q definitions.
3. **Calibrating on the wrong observable** (Q levels, ill-conditioned) and
   ignoring the 1D n_g≡n_eff level offset — the engine must be used as a shape
   ratio anchored on a measured row.
4. **The deleted width law** assumed a uniform-grating exponential envelope on
   APODIZED devices (and was validated circularly against void widths). The
   piecewise κ(z) engine drops that assumption → apodized widths now <1%.

## 8. Physics finding (new, zero-GPU)

The light-cone leak of the STORED envelopes reproduces the documented ranker in
the Q_i GROWTH phase (N=60-120), but leak keeps falling across N=165-195 while
measured Q_i is flat. **The Q_i saturation ceiling is therefore NOT
light-cone-limited** — another channel (3D/out-of-plane scattering, sidewall)
sets it. The saturating empirical fit stays primary for Q_i; the light-cone
model is a growth-phase envelope ranker only.

## 9. Parked for the user (needs a decision)

- **`git push`** — 9 commits sit on branch `add-claude-rules-skills`
  (e163c42 … a6b8ed5). Nothing uncommitted; push needs permission.
- **TE >1e5 hardening rows** (Itai device, N≈200/232 at adequacy numerics:
  3 nm window / 0.75 pm + `os.environ["TM_SIM_TIME_PS"]="4000"` set INSIDE the
  runner because `--option3` does not forward env vars). ~10-16 h each — not
  "short", so not dispatched. Only needed before relying on the >1e5 regime.
- **Phase 4 hybrid splice** (short-FDTD S-matrix core + CMT wings, in-repo
  reimplementation of the MATLAB `FDTDElement` idea): NOT triggered — the
  scalar decoration multipliers passed (B8, ±3%). Every stored `.mat` already
  carries `S11_complex`/`S21_complex`/`T_matrix` if it is ever wanted.
- **The MATLAB project's bugs** (section 7.2) — fixing that repo is the
  user's call; we did not touch it.
