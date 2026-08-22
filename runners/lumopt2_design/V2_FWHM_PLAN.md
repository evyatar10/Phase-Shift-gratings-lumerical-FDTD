# V2 PLAN — FWHM-safe re-optimization (researched + offline-validated 2026-08-21)

Study: lumopt2 corr-325 campaign v2. Author session: 2026-08-21 (STORM research
+ offline validation). Status 2026-08-21 (same day, later): **ENGINE IMPLEMENTED
+ W0 AND THE LOCAL HALF OF W1 PASSED** — `lumopt2_design.py` now carries
`CampaignSpec.width_grad` (+`wg_anchor/wg_mu/wg_lam_hi/wg_lam_lo/
adj_fix_field_re/im`), `soft_width_of_line`/`softw_and_weight`,
`width_band_penalty`/`make_fct_v2`, `make_width_classes` (WidthResults +
MixedFom with the weighted single-λ monitor-source adjoint), per-eval
`softw_um`/`fwhm_hat_um`/`wg_resid_um` logging with a `[WIDTH-GRAD SURROGATE
OFF]` alarm, and per-restart re-anchor + AL multiplier updates in
run_campaign. `width_grad=False` (default) leaves every existing spec
bit-identical. Local gates measured: golden fwhm 7e-15 µm; softW tracking
2.18 pp max; autograd-vs-FD 6.6e-7 (a detached-normalizer bug caught at
2.2e-3 and fixed — the softmax scale must stay in the graph); AL band values
+ signs exact; MixedFom constructs against the real local lumopt2; no-anchor
spec refused pre-build. NOTHING dispatched. Next: W1-remainder (toy
completion run + adjoint-source verification) and W2/W3 on cluster.
Read `HANDOFF.md` first (§0 non-negotiables). This file adds: (1) the verified
root-cause chain, (2) an offline-VALIDATED differentiable width observable,
(3) the v2 architecture, (4) the gate ladder for a one-shot campaign,
(5) the external-research digest (STORM method: 4 perspectives, full citations
in the session record).

---

## 1. Why the last campaigns failed — final, verified chain

All three links are MEASURED (sources: HANDOFF §2/§5, jobs 134334/134335, and
re-verified this session from the local `fspw_*_profile.npz` + stored `.mat`):

1. **Wrong observable class.** σ (2nd moment) cannot bound FWHM (level-set
   observable). Verified again offline this session: across the 7 corrected
   designs σ tracks FWHM growth with up to **24 pp** error (FWHM +26.6% reads
   as σ +2.5%). This is structural, not statistical: at fixed σ the envelope
   can flatten its core arbitrarily. The participation ratio (∫I)²/∫I² is
   equally blind (**21 pp** error, measured same test) — ALL global L²-type
   moments are excluded as width constraints, forever.
2. **Constraint-unit hole.** RHO_DN=0.95 in ρ units = +21.7% permitted FWHM in
   spec units (HANDOFF item 27). A band not converted to spec units is
   decorative. (Literature name for the whole failure: *Extremal Goodhart /
   proxy gaming* — the optimizer's job is to leave the calibration family.)
3. **Broken extraction.** `profile_line` y-row bug (fixed + validated 7e-15).

Corollary already in force (§0b): the adjoint MACHINERY is fine; the problem
SPECIFICATION was wrong.

## 2. The validated differentiable width observable — `softW`

**Definition (all autograd ops):** on the y-integrated resonance intensity line
I(x): smooth with boxcar(258 nm fringe period) then Gaussian (σ_k 0.25 µm) via a
precomputed convolution matrix → soft-max peak P, fixed-edge-window floor F,
half level h = F + 0.5(P−F) → `softW = ∫ sigmoid((I_s−h)/(ε(P−F))) dx`, ε=0.05.
As ε→0 this IS the floor-relative FWHM (co-area/IFT identity); at finite ε it is
the smoothed level-set width of shape optimization.

**MEASURED offline validation (this session, scripts in session scratchpad;
data: 7 corrected profiles `results_from_athena/fsp_width/`, 5-file nladder,
8-file shift_c400, 5-file tm_te_shift):**

| test | result |
|---|---|
| tracks fwhm_env growth, campaign designs (+4.9%→+26.6%) | max err **1.6 pp** (scipy form) / **2.2 pp** (autograd form, untuned) |
| cross-family: shift ladders (c400 ±, wide/narrow, TE/TM) | ≤ **2.1 pp** |
| cross-family: N-ladder (large excursion, N60 = −13% width) | err −8 pp → **local surrogate only; re-anchor every accepted iterate** |
| σ on the same tests | up to **24 pp** (blind — the control) |
| autograd gradient vs FD directional derivative | rel err **1.4e-8** |
| reference identity: our `fwhm_env` re-impl vs stored `fwhm_m` | **0.0** (and engine already validated 7e-15) |

Gradient structure (verified): concentrated at the two half-max crossings plus
distributed peak/floor terms — exactly the implicit-function-theorem FWHM
derivative, mollified. Plateau-safe (no 1/|I′| blow-up).

**Definition-identity audit (2026-08-21, user asked; MEASURED):** the engine's
`fwhm_env_um` chain vs `post_processing`/`extract_and_process_field_profile`,
step by step — λ-index pick: same argmin, resonance from the same shared
`find_bragg_resonance`; E-slice ndim handling: identical code; intensity:
|Ex|²+|Ey|²+|Ez|², identical; y-integral: trapz over y, identical; envelope +
FWHM: the SAME imported `sim_helpers` functions (not copies). ONE nominal
difference: crop — project uses `x_grating_end` = N·pitch + cavity/2, engine
uses N·pitch (0.13 µm tighter/side). Measured on 4 stored families (N100/N80
c325, c400 shift, TE/TM shift): FWHM difference **0.0000 nm** in every case
(the extra span is flat floor; envelope peaks/floor/crossings untouched).
The conventions are numerically identical; production confirms additionally
go through `post_processing` itself. σ/softW are NOT the spec observable —
softW is only the gradient carrier, anchored to `fwhm_env_um` per restart.

**softW is a GRADIENT CARRIER only. The authority is always the measured
`fwhm_env_um`** (per-eval, logged, WidthTrip band, restart filter — already
shipped). softW is re-anchored to the measured value at every accepted iterate
(constrain the *increment* of softW against the *measured* absolute) — this
kills the item-24 basin-transfer failure by construction.

## 3. Adjoint feasibility in lumopt2 R1.3 (read from local source this session)

`fom/field_fom.py` + `fom/base_fom.py` (local install, v261):
- Stock `FieldFom` supports only per-λ scalar Σ|E|² FOMs; its adjoint source is
  hard-coded `conj(E_fwd)` on the monitor. **An arbitrary spatial functional
  needs a subclass** whose `setup_adjoint_simulation` imports the WEIGHTED
  source `W(x,y)·conj(E_fwd)`, with `W(x,y) = [dF/dI(x)]·(y-trapz weight)`
  computed by autograd.grad of §2's softW on the forward profile (numpy, per
  eval). One extra adjoint solve then yields the exact functional gradient
  (adjoint superposition); the per-λ jacobian bookkeeping reduces to identity
  at the (stop-gradient-selected) resonance index — same detached-index
  pattern as the T softmax window.
- Mixed port+field FOM: `calculate_gradient_fields` already loops entries and
  sums; a `MixedFom` dispatching per sim_result type (Port entries → port
  logic, Field entry → weighted-source logic) follows the existing
  BoundaryCorrected-subclass pattern. **Cost: +1 adjoint per iteration**
  (≈26 → ≈39 min per gradient on H200).
- ★The field-adjoint path has NEVER been FD-validated in this program and the
  port path needed the complex C-fix (skill item 6). **Assume the field path
  needs its own C_field, measured by the same recipe** (naive + adj-only(0,1)
  + grid fit), gated before any campaign (gate W3 below).

## 4. V2 architecture (synthesis of all four research perspectives)

**★Basis note (2026-08-21, user):** "SPINS" was a mistranscription — STORM
(the research method) was meant. The architecture below is chosen on GENERAL
grounds only; SPINS is retained in §6 merely as one literature datapoint.
Why augmented Lagrangian is the best general option here (re-derived without
any SPINS reference): (1) it is the standard remedy for exactly our failure —
fixed-weight penalties either leak (weight too small) or ill-condition the
Hessian (too large), while AL drives violation → 0 via multiplier updates at
finite μ (Bertsekas 1982; Nocedal & Wright ch. 17; Gramacy et al. 2016 for
expensive blackbox constraints; standard practice in stress-constrained
topology optimization); (2) the multipliers can be updated on the MEASURED
violation — the truth steers feasibility, the surrogate only steers inner
gradients; (3) it needs NO optimizer replacement — L-BFGS-B (lumopt2's only
optimizer) stays, so every measured operational behaviour (bounds-as-trust-
region item 22, scaler round-trip item 21, restart semantics item 13) stays
valid. Ranked alternatives, banked as FALLBACK: NLopt CCSAQ/MMA or scipy
trust-constr with the width as an explicit inequality (more robust in theory,
but requires a ScipyOptimizer subclass + separate ∇T/∇W plumbing, and
invalidates the L-BFGS-B-calibrated trust machinery). TRIGGER for the
fallback: multipliers oscillate or measured feasibility fails to improve
across 2 consecutive outer cycles in W4/W5. Fixed-weight penalty and
SLSQP are rejected outright (leak-or-ill-condition; noise-intolerant).

FOM (maximize): `J = softmaxT_window − λ·g − (μ/2)·g²` — augmented Lagrangian
on the width-band violation `g` of delta-anchored softW (asymmetric band
+2%/−5%, the user's standing spec), inner-solved by the existing L-BFGS-B:

- **AL multipliers updated between stages on the MEASURED fwhm_env violation**
  (the truth updates λ; the surrogate only steers inner gradients). Expected
  4–8 outer × 10–25 inner iterations (literature estimate).
- **Keep:** trust_nm on every free block (item 22), recenter policy, cold-start
  resume, per-eval profile .npz, WidthTrip on measured fwhm_env + restart
  filter (`fwhm0_um`), elongation wall, comb pre-computed not co-optimized
  (platform recipe), physics-informed seed **+ see-saw seeding** (§6e:
  Δcorr=±δ, Δavg=±δ/2 on teeth 1-2 — the optimizer provably does not discover
  it from a smooth seed).
- **Filter acceptance** (Fletcher–Leyffer): an accepted-best must improve FOM
  or reduce measured band violation — never worsen both.
- **Measured re-trim (projection)** at stage boundaries + final delivery:
  bisect ONE scalar knob on measured fwhm_env back to the target width, then
  score T (HANDOFF §6 scheme; forward-only, 2-3 sims). **Every reported T is
  at the re-trimmed width; never compare T across widths** (§0d).
- **Family monitor:** log fwhm_env/σ ratio per eval; drift ⇒ proxy-gaming
  signature ⇒ inspect before trusting further steps (alarm threshold re-derived
  from y-integrated data, replacing the retired 0.978).
- ρ-penalty: retire as a constraint; keep 2κL ≥ 3.5 hard guard.
- DROPPED, do not reintroduce: σ band as the width control (blind), fitted
  param-space FWHM slopes (user-deleted), CMT width models (user-deleted),
  raw-line FWHM (user-deleted).

## 5. Gate ladder for a ONE-SHOT campaign (each gate is a hard stop)

| gate | where | what | PASS |
|---|---|---|---|
| W0 | local, 0 GPU | golden-file: fwhm_env on stored .mat ≡ fwhm_m; softW tracking table (§2); autograd-vs-FD of the fct | done this session (re-run after any engine edit) |
| W1 | local, 0 GPU | MixedFom build smoke; bounds/dp clamp (item 20) exercised through ONE gradient computation; guard fault-injection through the double-wrap (item 11); **2-eval toy run reaching the completion path** (item 23) | all paths exercised |
| W2 | cluster, 1-2 fwd | canary: seed's measured fwhm_env + T + λ vs in-study anchors (cite stored; no control re-runs) | anchors reproduced |
| W3 | cluster, ~18 sims | **FD gate for the WIDTH gradient** at a detuned in-band point: naive validate_gradient (fd first), adj-only (0,1) run, fit C_field; per-class residual ≤ few %; re-verify at a 2nd operating point | 6/6 sign, α∈[0.8,1.25] after C_field |
| W4 | cluster, mini | known-answer mini-opt: start from BEST_T9635 (measured 20.34 µm), AL loop must walk width back into band while keeping T > noshift's 0.9345-level trade | recovers band, T above shift-free line |
| W5 | cluster | THE campaign (both-seed parallel per standing rules) | — |
| W6 | cluster | close-out: scale-check ladder, decoration-removed row, production confirm N≈165-169 accurate mesh OUTSIDE lumopt2, lock-target re-trim; state mesher for every number (PVA≈0.92×conformal) | §2-sane |

Pre-dispatch (unchanged, mandatory): license seat probe from IGUM, quota check,
h5_roll_clean, QOS 4d_1g for the driver, per-study sweep list, job ID reported.

## 5a. Recording λ-window: ADOPT ±5 nm @ 20 pm (501 pts) for v2 (user, 2026-08-21)

The banked wider-window candidate (skill "Future-campaign candidates") is
adopted for the v2 campaign. Decision criterion recorded 2026-08-16 was "adopt
if recenter churn does not decelerate" — it did not (MEASURED: λ drifts ~+1 nm
per accepted early iteration, probes jump +2.6 nm, three jobs died at the band
edge in one day before the gen-3 policy, and stage restarts remained frequent
through every campaign). v2 runner values: `scan_width_nm = 10.0` (±5 nm),
`n_wl_points = 501` (SAME 20 pm grid — resolution unchanged, span wider).
Engine defaults stay 6.0/301 (legacy specs bit-identical). This is a NAMED §2
numerics change ⇒ the W2 canaries anchor the v2 stack fresh (they were
required anyway, so the adoption is free); never compare v2 absolute T to
±3 nm-window rows without noting the window. RAM: monitor-driven, 501-pt
profile monitor ≈ 1.7× — well inside the measured 160G headroom.

## 5b. PROOF-OF-GAIN protocol (user, 2026-08-21: "validate that we actually
## get the increase we wanted — not optimizing non-helpful stuff")

The campaign's claim is "T rose at the SAME measured FWHM". Every layer that
must independently support it:
1. **Before the campaign — the §6b decisive experiment** (HANDOFF): re-trim
   the banked best to `noshift`'s width by bisecting corrugation on the
   MEASURED fwhm_env (2-3 forwards) and compare T against the shift-free
   0.9345 at that exact width. Settles whether the old campaign's discovered
   direction contains ANY real constant-width gain before v2 spends GPU.
2. **W4 pass criteria (hard):** from BEST_T9635 (20.34 µm, over-band), the AL
   loop must (a) return the measured fwhm_env into the band, (b) finish with
   T at or above the constant-width interpolation line of the shift-free
   family, (c) multipliers settle (no oscillation over 2 outer cycles).
   Anything less = cost function not proven; do NOT proceed to W5.
3. **In-campaign non-helpfulness tripwires:** marginal efficiency dT per µm
   of measured width spent, logged per accepted iterate (platform recipe:
   stage is DONE when it collapses ~100×); the fwhm_env/σ family monitor;
   the wg residual alarm. A campaign that only "gains" while any width
   metric climbs the band is optimizing the forbidden lever — stop.
4. **Close-out (W6, unchanged but restated as the proof):** the winner is
   re-trimmed to the ORIGIN's measured width before ANY comparison (never
   report T at a different width — §0d); then decoration-removed row, shift
   scale-ladder ×0/×0.5/×1.5, production N≈165-169 + accurate mesh, and a
   conformal-mesher cross-check. Only the re-trimmed, production-confirmed
   number is reportable.

## 6. External research digest (STORM, 2026-08-21) — what transfers

- **SPINS (Stanford, Su et al. APR 7, 011407 (2020); arXiv:1910.04829)** —
  BACKGROUND ONLY (the user meant STORM, the research method, not SPINS; the
  v2 design does NOT rest on SPINS): staged "optplan" + L-BFGS-B inner + an
  augmented-Lagrangian wrapper + penalty continuation; fabrication
  constraints as analytic differentiable penalties (Vercruysse Sci. Rep. 9,
  8999 (2019)). Its value here is only corroboration: an independent
  production framework converged on the same AL-over-L-BFGS-B shape that the
  general literature prescribes. No mode-width constraint exists in SPINS.
- **Meep/Johnson epigraph route:** any smooth secondary observable rides as an
  NLopt CCSAQ / scipy trust-constr inequality; constraint scheduling (activate
  late) is standard. Fallback if AL misbehaves (needs optimizer subclass —
  banked v2 idea in skill item 22 already).
- **Target-mode-overlap FOMs** (Lalau-Keraly OE 21, 21693 (2013); Lu&Vučković
  OE 19, 10563 (2011); Portalupi OE 18, 16064 (2010)): the literature's way to
  pin SHAPE. Rejected as PRIMARY here (over-constrains: the spec is a width
  band, not a shape lock; and complex-field overlap is fringe-phase-stiff) —
  keep as a weak secondary penalty only if a new cheat mode appears.
- **Precedents for our exact failure:** stress-constrained topology
  optimization re-normalizes aggregate constraints against the true local max
  EVERY iteration (Le et al. 2010 "adaptive constraint scaling") — our
  re-anchoring is that, 15 years late. Trust-region surrogate management
  (Alexandrov-Lewis) requires first-order consistency at every re-center.
  Noisy-constraint SQP: never adjudicate feasibility below the measured noise
  floor (ours: 0.03% width, HANDOFF §4).
- **Physics leads surfaced (NOT dispatched — candidates for the user):**
  1. **Sinc-envelope loophole** (Sauvan/Lalanne OE 12, 458 (2004)): a sinc
     envelope has top-hat k-spectrum → ~zero light-cone weight at FINITE
     intensity-FWHM; the spec constrains FWHM, not energy — weak oscillatory
     κ(x) tails (κ sign changes) are the theoretically optimal family for
     "high Q at fixed FWHM". Give the corr basis room for sign-changing
     deviations; check the seed's reach.
  2. **Arms/ends radiate most** (Kazarinov-Henry + Henry 1988 measurement:
     radiation cancels at center, concentrates at grating ends) — matches our
     measured "~70% in the arms"; per-region comb phase at the ARM ENDS is the
     least-explored comb axis.
  3. **BOX/under-cladding thickness as λ/4 radiation-recycling mirror** (Mock
     JLT 28, 1042 (2010): Q oscillates with reflector gap, max at (2m−1)λ/4)
     — width-neutral, never examined in this program.
  4. **TM→TE conversion loss never itemized** — if part of the 16-19% loss is
     polarization conversion, the remedy is parity, not envelope shaping.
  5. **Cladding-modulated grating strength** (APL 123, 191106 (2023), TM):
     move part of κ from sidewall teeth (the radiators) into the cladding row.
  6. CMT leverage at −3 dB lock: T = (Q_tot/Q_wg)² ⇒ every % of Q_i ≈ 2× its
     naive weight in T near critical coupling (DERIVED).

## 6b. Zero-GPU light-cone model — VALIDATED RANKER + what it says (2026-08-21)

Built on user request ("think outside the box; try things without GPU but
validate them"). Model: leakage = light-cone weight of the envelope's
k-spectrum (Englund picture), computed from STORED measured envelopes.
**Calibration (MEASURED inputs): log-log correlation 0.975 against the
measured Q_i N-ladder (N60-120)** — the model RANKS correctly. Slope 0.32 ⇒
COMPRESSIVE: real-world gain ≈ (model ratio)^0.32. A ranker, never a Q
predictor. Script: session scratchpad `lightcone_model.py`.

Findings at FIXED intensity FWHM 19.245 µm (all DERIVED from the validated
ranker; measured-envelope N100 = baseline):
| candidate | model leak vs baseline | compressed (^0.32) | verdict |
|---|---|---|---|
| exponential (= measured shape) | 1.017× | — | sanity check ✓ |
| **Gaussian envelope** | **0.008× (~125× less)** | **~4-5× Q_i** | THE headroom; agrees with the independent §6d model (~105×) — two models converge |
| **Gaussian core reachable with 25 free teeth** (exp tail past ±12.9 µm) | **0.063× (16×)** | ~2.4× Q_i | available INSIDE the current campaign basis |
| Gaussian core with **40 free teeth** | 0.0062× (~160×) | ~5× | captures the full gain; saturates (50/100-free no better) — third independent convergence with §6d-(iii)'s taper-40 plateau |
| sinc / sign-flip κ | 3.7× WORSE | — | needs device ≫ mode (many lobes); at our span it backfires — DEMOTED |
| flat-top | 2.7× worse | — | sharp shoulders radiate |
| quadratic chirp (arm radiation phase) | −9% at best, ≥6× worse beyond | — | chirp spreads the spectrum INTO the cone — DEMOTED (user's hunch confirmed; consistent with closed distributed-shift) |

**Outside-box STORM sweep (2026-08-21, agent + model adjudication):** ranked
candidates with status — (1) distributed Kazarinov-Henry arm-phase chirp:
**CLOSED by model optimization** — under the width-protecting slope cap
(|dφ/dx| ≤ 0.4κ) the best feasible designed profile gains nothing (best
cap-riding 0.73× model ≈ 10% Q_i compressed, at width risk); (2) **second
comb row per side (Noda double-lattice pattern) — the live candidate**:
independent (δx₂, y₂) gives the second complex amplitude to null the grazing
lobe; zero-GPU check = two-row superposition on the response-matrix program's
measured single-row data; then ONE discriminating FDTD; (3) BOX-thickness λ/4
recycling (wafer choice; transfer-matrix check first); (4) phase-matched
cladding counter-corrugation (FBG "equal photosensitivity" import); (5) SWG
shroud (piggybacks on the TM→TE conversion audit, which the literature
supports via "magic-width" lateral-leakage suppression). Stacking law from
PCSEL/QNM literature: cancellation channels combine as COMPLEX AMPLITUDES —
adding any second canceller requires re-tuning the comb's δx (matches our
measured sign-inversions). Full citations: session record (agent report).

**Two-row comb phasor model (2026-08-21, zero GPU — PRELIMINARY):** the prior
"second row" negative (scat_c2_row2, job 121392, 2026-07) does NOT transfer:
it tested NEAR-FIELD rows at y=700/900 (evanescent-decay-dominated ×0.5 per
250 nm + a 40 nm-gap dimer artifact), while the comb operates in the
FAR-FIELD grazing-lobe regime at d=1.9 µm where standoff is measured LOOSE.
Phasor fit (1/Q_i = S|1+a·e^{iφ(δx)}|², exactly determined on the 3 measured
comb_q3db points): a = 0.165 — independently consistent with the archive's
"passive canceller phase-limited ~13%" note; two anti-phased rows ⇒ ceiling
Q_i ≈ 104k (2.2× ctrl) [DERIVED]. Model also predicts the single-row optimum
at δx≈315 (−58° from the 401 winner — never sampled; the basin scan sampled
+90/+180/+270 only) AND a tension: it rates the +270° basin point BETTER
than the winner while the scan measured all ≤ — different device family
(N165 conformal fit vs campaign device), resolvable FREE by refitting on the
stored basin-scan per-row numbers (fetch at next server contact). Decision
path: refit → if coherent, a 4-6-point δx fan (cheap forwards) settles the
single-row optimum + overdetermines the ceiling BEFORE any two-row FDTD.
Stacking rule from the QNM/PCSEL literature: re-tune δx jointly with any
second canceller.

**Consequences for v2:** (1) the FWHM-fenced campaign has real, large headroom
in exactly the direction the optimizer can reach — Gaussian-core apodization
with the re-trim holding FWHM (needs κ RISING outward: corr ramp low-center →
high-outer, in-bounds 150-500 with mean-ρ compliant); (2) ★CANDIDATE ENGINE
CHANGE, post-W-gates: `N_FREE` 25 → 40 (+30 params) captures ~10× more model
gain — new bounds/FD gates required, so it is a v2.1 decision, not mid-gate;
(3) sinc and chirp are closed cheaply, without GPU. CAVEAT: scalar 1D model,
κ→envelope map idealized; the §6f fab-ceiling question still bounds everything.

## 7. Honest expectations

The only validated fixed-width gains in the program: comb +0.0119 T / +17.1%
Q_i, see-saw −31% loss (corr-400, unported), air trench, W1250 cavity — all
≤1-2% effects vs the sigma-era +7 T-points that were width-bought. A correctly
fenced campaign should be expected to land WELL below 0.96; origin is 0.893.
Success = genuine T gain at measured-constant fwhm_env, delivered at the
re-trimmed width, confirmed at production N + accurate mesh.

## 8. GATE LOG (running)

- **W2 PASS (MEASURED, job 135971 task 10, 2026-08-21):** v2 canary through
  the full width_grad stack at the ±5 nm/501 window. T 0.8905 (anchor 0.8912,
  window delta as expected), λ 1564.274, Q_L 2039. **Campaign anchors:
  wg_anchor = {softw: 18.160689, fwhm: 17.713551}, fwhm0_um = 17.713551**
  (softw = the single-λ twin's own sample; broadband softW 18.1709 → twin
  consistency 0.010 µm; fwhm_env vs stored-fsp recovery +0.013 µm = the
  window/numerics delta, now anchored). Written into validate_c325 PROV_*.
- W1r/W3a (tasks 11/12) failed on the pre-fix code (port-loop bug, skill item
  28d) — resubmission with the fixed engine after task 13 drains.
- Basin-scan per-row recovery (from job 133718 logs): pitch524 T 0.94374 /
  pitch540 0.94165 / r70 0.94591 / r100 0.94547 / d1700 0.94587; anchors
  comb 0.94629, no-comb 0.94147 (campaign device, N100 PVA). Phase rows 0-2
  pending one grep — then the phasor refit.
- **Two-row comb — DOWNGRADED by the basin phase fan (MEASURED, job 133718
  rows 0-2 + anchors, campaign device N100 PVA):** the 4-point ±symmetric fit
  separates coherent from incoherent: coherent amplitude a≈5.5% (not 16.5% —
  the N165 3-point fit could not separate the terms and over-read it), winner
  phase within ~2° of optimal (basin verdict CONFIRMED with mechanism; the
  "δx 315" tension was a wrong-family artifact, withdrawn), and mis-phasing
  adds a large phase-INDEPENDENT (incoherent scattering) loss term. Two-row
  optimally-phased ceiling on this device ≈ −16% loss (~+8% Q_i) [DERIVED] —
  at most one cheap test someday, NOT a priority. Phasor projections must
  always include a ± symmetry pair to separate the incoherent term.
- **SEED AUDIT (MEASURED, job 135989 t0/t1, 2026-08-22 — answers the user's
  seed-A/B question):** seedB dip seed FWHM 18.1430 µm = **+2.50% vs origin
  17.7005 — over the +2% band AT BIRTH** (void-σ read it as NARROWER than
  origin: ordering inverted, not just missed). seedB best (T 0.9468):
  FWHM **20.1630 = +13.9%**; seedA best 20.3362 = +14.9%. ⇒ "A converged
  toward B" was convergence in width-violation: both lineages bought T with
  ~+14% width; σ hid it identically (+1.3%/+1.6%). CONSEQUENCE for v2: the
  dip seed is NOT usable as-is — v2 seeds must be measured-in-band at start
  (origin + see-saw seeding stands; any dip-profile seed needs a re-trim to
  the band first). Rebuild-from-params verified EXACT (T to 1e-6).
- **FieldRegion adjoint SURVIVED (136026, all 3 tasks past the old crash
  point, no errors)** — first field-adjoint solves in program history.

## 9. USER DECISIONS 2026-08-22 (verbatim intent, recorded same session)

- **Dip seed: OUT as-is** ("2.5% is just over our definitions — too much").
  Optional second seed = the dip profile RE-TRIMMED into band by bisecting
  its amplitude on the MEASURED fwhm_env (2-3 forwards, §6 projection
  pattern). One-seed vs two-seed decision PARKED to W5 dispatch;
  recommendation: origin+see-saw primary; band-retrimmed dip as second seed
  (two independent basins were this program's own convergence evidence).
- **Comb: NO binary existence/count parameters in v2** (user: don't add
  confusing parameters; the count was measured flat 29-113 and the basin
  scan confirmed the winner optimal in every direction). v2 runs the comb
  FIXED at the winner geometry (sliver-freeze mechanism); if any spec frees
  it, log comb drift vs seed per accepted iterate and expect ~motionless
  (platform recipe). Comb-only optimization deferred to a later dedicated
  stage if ever wanted.
- **GATE CALIBRATION (user, 2026-08-22, verbatim intent): "it's not always
  about verification — start optimizing and seeing improvement IS the way."**
  Adopted: W3's C_field fit stays (data already in flight, zero extra cost);
  **W4 is FOLDED INTO the campaign start** — the campaign dispatches right
  after the C_field fit, and its first ~3 evals are the live gate (healthy =
  measured T rises or width falls at in-band measured fwhm_env; the
  WidthTrip/filter guarantee means a bad gradient wastes hours, never
  correctness). The §6b retrim (136051) supplies the known-answer info W4
  would have. Campaign seed: origin + see-saw (single seed first; second
  basin later if wanted).
- **Taper-shape research (2026-08-22, full digest in session record):** the
  fixed-HALF-MAX-WIDTH optimal-κ(x) problem is ABSENT from the literature
  (published variants fix footprint/energy/V; WBG-cavity papers never track
  mode size) — our numerical solve of it is novel. Adopted design rule with
  published backing (Oskooi OE 20, 21558): the ramp→mirror knee must be
  C¹-SMOOTH. Shape family for v2 seeding: κ(x)=κ_m[a+(1−a)(|x|/L_t)^p],
  optimize (a,p,L_t) at fixed FWHM offline. κ-below-150nm-teeth fallback:
  misaligned-sidewall κ=κ₀|sin(πΔP/Λ)| (measured on SOI TE; on OUR closed
  list as a global knob — reopen only with a TM/SiN calibration sim and the
  §6f burden-of-proof note). TM κ(depth) calibration for SiN doesn't exist
  publicly — ours is novel too (writeup material).
- **★NOVEL RESULT (2026-08-22, zero GPU): the fixed-FWHM optimal envelope,
  solved numerically in the calibrated ranker** (8-knot log-intensity spline,
  FWHM pinned 19.245 by bisection): leak = 0.00073× measured (Gaussian
  0.008×, 25-teeth-reachable 0.063×). Shape: Gaussian-like core, mid-tails
  HEAVIER than naive (45% energy beyond ±13 µm vs 77% measured), far tails
  FASTER-than-Gaussian (I/I0 0.039@20µm, 1.9e-5@32µm). Compressed (^0.32)
  ≈ 10× Q_i at full reach — needs κ shaped over most of the device ⇒ the
  quantitative case for N_FREE 25→40(+): reachable-25 stays ~2.4×. Knot
  table in session record (scratchpad optimal_envelope.py output). This
  optimum + the C¹-knee rule define the v2.1 seed/target shape.
- **Mesh-integrity check (user asked, 2026-08-22; MEASURED from stored
  profiles):** solver x-grid BIT-IDENTICAL across all 7 different-geometry
  solved runs (2068 cells, uniform 49.989 nm) — fixed grid, geometry moves
  across it, comparisons valid. Note: pitch/dx = 10.34 is INCOMMENSURATE, so
  "cell edges touching tooth edges" never held exactly in any era; the
  protections are fixed-grid + in-study controls + the measured 0.0018
  half-cell jitter floor + PVA's smooth sub-cell response + forced-symmetric
  z mesh.
- **★CORRECTION (user caught it, 2026-08-22): the REGULAR pipeline's mesh is
  PITCH-LOCKED by design** — bragg_device dx_override = pitch/10 (51.683 nm
  for TM 516.83), comment: "places mesh edges exactly on the wide/narrow
  transitions". My claim that "exact edge-touching never existed in any era"
  was WRONG — it is the regular pipeline's deliberate design, and dx followed
  the pitch when the pitch changed. The CAMPAIGN's lumopt2 region hardcodes
  50.0 nm (its "matches the global mesh" comment is true only for pitch-500
  TE) — so the campaign both breaks edge alignment and mismatches the
  device's own override. Campaign-internal comparisons unaffected (one fixed
  grid + in-study anchors); part of the campaign-vs-family offset is now
  mechanistically attributed. ★v2.1 CANDIDATE (named §2 change, needs fresh
  W2 anchors — do NOT apply mid-stream): set the campaign region dx to the
  pitch-locked 51.683 nm. Also: my earlier jitter-floor argument was
  mis-applied (the 0.0018 floor measures displacement sensitivity OF the
  aligned config, not evidence against alignment).
- **★IDENTITY TEST PASS (MX task 14, job 136077, MEASURED):** lumopt2 stack,
  bare, conformal + pitch-locked 51.683 → λ 1559.024 / FWHM 19.2493 vs
  stored regular anchor 1559.006 / 19.2448 — **18 pm and 0.02%**. lumopt2 ==
  regular physics; the historical lens offset is FULLY attributed (50.0-nm
  misalignment + PVA). T +0.0079 (the numerics-sensitive one, as §2 says).
- **★RETRIM VERDICT (136051, bisection converged):** d+52.5 → 17.755/0.9597,
  d+56.2 → 17.637/0.9586 ⇒ at 17.7005: **T ≈ 0.959 vs origin 0.8926 =
  +0.066 T at equal width — the §6b answer is YES** (surrogate, PVA-50
  pipeline; MX-16 cross-checks at corrected mesh; production confirm still
  the reportable step). Width payback cost ≈ 0.002 T/µm (uniform corr-add).
- **MX-GRAD dispatched (task 18, job 136090):** tooth-gradient FD at
  pitch-locked+conformal — if alignment cures the staircase alphas, v2.1
  campaigns run IN the production convention (no mesher split at all).
- CPU width-adjoints (136046): >7 h each and still solving — in-loop width
  gradient effectively priced out for campaigns; projection architecture is
  the recommendation regardless of the C_field outcome (fit still runs when
  vectors land, for the record and for occasional calibration use).
- **★ARCHITECTURE SETTLED BY DATA (2026-08-22):** CPU width-adjoint MEASURED
  at 31,259 s (8.7 h) — in-loop width gradients are priced out. **v2.1
  campaign = PROJECTION-FIRST**: FOM = softmax T (+ cheat-channel walls),
  measured fwhm_env guard, re-trim projection at stage boundaries (proven:
  ~5 forwards, 0.002 T/µm), restart filter on the band. width_grad stays an
  occasional-calibration tool. Bug fixed post-hoc: MixedFom lacked
  dipole_base_amplitude (task 11 died AFTER its adjoint — item-23 class);
  tasks 12/13 will fail the same way but their SOLVED adjoint .fsp files are
  the artifact — a contraction-only pass recovers the C_field vectors with
  zero re-solving. RETRIM VERDICT sealed: T 0.95916 @ 17.695 µm (+0.0665 at
  equal width); MX-16: survives corrected mesh (+0.060 T at 2.7% NARROWER
  than corrected origin). MX-15: the 50-nm misalignment distorted origin
  width +3.6%; aligned PVA-vs-conformal gap −4.7% (was −8%). Campaign spec
  finalization waits on MX-GRAD (mesher) + MX-17; seed = the re-trimmed best.

## 10. LIVE STATE + DECISION RULES (Fable handoff, 2026-08-22 — execute as written)

**Running:** 136104 = THE v2 projection campaign (4d_1g; seed = retrimmed
best; resume = re-dispatch same module). 136090 = MX-GRAD. 136077_17 =
rho15 at corrected mesh. 136046_12/13 = will FAIL after their ~9 h adjoints
(known bug, fixed in repo) — do NOT redispatch them; their solved adj .fsp
under results/validate_c325/.../w3fd|w3quad_files/ are kept for a future
contraction-only C_field recovery (recipe: load fwd+adj in a task, call
project.fom.calculate_gradient_fields + parametrization contraction — needs
engine work, NOT urgent; projection campaign doesn't use C_field).

**Campaign health (judge at every wake, first 3 evals especially):**
- HEALTHY = accepted rows with T ≥ 0.9597-ish rising OR fwhm_env falling;
  fwhm_env/17.7136 within [0.95, 1.02] on accepted-best; no repeated
  ABNORMAL_TERMINATION_IN_LNSRCH.
- WidthTrip/RecenterNeeded = engine restarts by itself (resume-safe);
  only report.
- STALL (fom Δ < 1e-4 over 5+ evals) or NEW error signature → stop branch,
  fetch log, escalate to Fable.
- Fetch-early every wake: evals.jsonl (KB) → local results_from_athena/.
**MX-GRAD verdict rule:** tooth-class α no longer 10-30× off (sign correct,
spread ≲ 3×) ⇒ conformal+pitch-locked gradients usable ⇒ at the NEXT
campaign anchor reset (stage boundary), migrate spec to region_dx 51.683 +
conformal + fresh W2-class anchor (ONE canary), eliminating the mesher
split. Ambiguous ⇒ park for Fable.
**MX-17 rule:** just record rho15's corrected-mesh row next to the others.
**Stage boundary (campaign plateau or trip):** run the retrim bisection
(retrim_best_c325 pattern) on the stage best to fwhm0, log the (fwhm, T)
curve, re-seed next stage from the trimmed vector. Never compare T across
widths. **Production confirm (needs user go):** winner at N≈165, conformal,
accurate mesh + lock-target re-trim + comb δx re-check.

## 11. MESH-CORRECTION REVALIDATION — COMPLETE (job 136077, MEASURED)

All four rows at the CORRECTED pitch-locked mesh (dx = pitch/10 = 51.683 nm):

| row | T | FWHM µm | vs corrected origin |
|---|---|---|---|
| MX-14 identity (bare, conformal) | 0.9183 | 19.2493 | vs stored 19.2448 = **+0.02%** |
| MX-15 origin (comb, PVA) | 0.8996 | 18.3460 | — (the corrected anchor) |
| MX-16 retrim d+52.5 (PVA) | **0.9594** | 17.8530 | **−2.69% (NARROWER)** |
| MX-17 rho-neutral a=1.5 (PVA) | **0.9293** | 18.5013 | **+0.85% (in band)** |

**BOTH GAINS SURVIVE THE MESH CORRECTION:** retrim **+0.0598 T while 2.7%
NARROWER** than origin; rho-neutral **+0.0297 T at +0.85%** width. The
50-nm misalignment had distorted widths by +3.6% (origin), +0.6% (retrim),
+2.6% (rho15) — i.e. it inflated the apparent origin width most, so the
corrected numbers are if anything MORE favourable. λ identity 18 pm.
⇒ The +0.06 T-at-equal-width result is NOT a mesh artifact.

## 12. LIVE STATE 2026-08-23 (supersedes §10's job table)

- **136107 = THE CAMPAIGN** (v2proj, 4d_1g). Width protection = THREE layers,
  all verified armed: (1) `fwhm_wall` in the FOM gradient (measured slopes,
  re-anchored to measured fwhm_env at every accepted-best); (2) WidthTrip on
  accepted-best at band [0.95, 1.02]×17.713551 = [16.83, 18.07] µm — seed at
  17.755 (ratio 1.0023, in band); (3) `_best_from_log` filters restarts AND
  final selection on the same band. sigma tripwire OFF by design (retired).
- **136108 = THE GPU QUESTION** (import-source injection, GPU lane). Only the
  FieldRegion route is proven GPU-blocked; this tests whether ANY arbitrary
  field-sheet injection is blocked. PASS ⇒ exact FWHM gradient at GPU speed
  ⇒ reopen in-loop width gradient at the next stage boundary. FAIL ⇒ two
  independent mechanisms blocked; next discriminator (only if wanted) is a
  CROPPED source sheet (±20 µm captures nearly all width-gradient weight,
  which concentrates at the ±8.85 µm half-max crossings) to separate
  "feature unsupported" from "sheet too large for the launch config".
- **136090 = MX-GRAD** (conformal+pitch-locked tooth gradients; ~6-9 h).
- 136046_12/13: will fail at the end (known, fixed in repo); their solved
  adjoint .fsp are kept for offline C_field recovery.
- **QUOTA INCIDENT 2026-08-23:** job 136090 (MX-GRAD) killed by "Disk quota
  exceeded" — the roll-cleaner's glob only covered campaign_c325_* and missed
  every study of this session (skill item 29). Freed 132 G of stale
  *_output.h5 (>2 h old only; .fsp/logs/jsonl untouched), quota 330 G -> 199 G,
  cleaner v2 (all dirs, newest-2, 30-min guard, 15-min loop) deployed and
  running. Campaign 136107 and GPU test 136108 SURVIVED (verified). MX-GRAD
  re-dispatch HELD until the campaign has headroom — its 14 concurrent FD
  legs are ~49 GB of scratch and are what tipped the quota; when re-run, use
  4 indices, not 7.
- **CPU width-adjoint SECOND measurement: 43,537 s = 12.1 h** (136046_13;
  first was 8.7 h). Range 8.7-12.1 h/solve confirms the CPU route is
  unusable in a loop. GPU forward on the same node = 3,100 s (51.7 min), so
  a working GPU width-adjoint would be a 10-15x speedup and would make the
  in-loop exact FWHM gradient viable (+~50% per iteration instead of +500%).
  This is why task 19 (import-source on GPU) is the decisive experiment.

## 13. PROACTIVE BUG AUDIT 2026-08-23 — four latent defects in the RECOVERY path

Found by auditing the code that only runs after a preemption or a width trip
(item-23 class: least-tested code). All four FIXED on disk and covered by a
new local gate (scratchpad `restart_path_gate.py`, 4/4 PASS):

| # | Defect | When it would bite | Consequence |
|---|---|---|---|
| A | `fw_anchor` never re-anchored on a COLD restart — module re-import resets it to the SEED literal while `_best_from_log` resumes elsewhere | any preemption/requeue | width wall predicts at the wrong operating point (item-24 false-rejection class) |
| B | WidthTrip handler caps corrugation DOWN (`corr_max *= 0.95`) | any width trip | blocks the payback direction that FIXES the width — fights its own recovery |
| C | Selection filter FAIL-OPEN: a row whose profile extraction failed carries no width evidence and PASSED | transient monitor failure | an unmeasured (possibly over-wide) design could be selected/delivered |
| D | `fwhm_wall=True` + `fw_anchor=None` silently fell through to elongation-only | mis-specified runner | ALL width steering vanishes from the gradient, silently |

★**NOT IN EFFECT for the running campaign 136107** (loaded before the fixes —
item 15). Forward progress is unaffected; only the RECOVERY paths carry the
defects. **Recommendation: restart 136107** (user-approved scancel; it is
early — iteration 1 — so the cost is ~2 h and the whole recovery path becomes
correct). Rollback exposure is benign (C rejects only unmeasured rows).

## 14. ★★★GPU WIDTH-ADJOINT PROVEN (job 136108, 2026-08-22) — the limitation was
## the OBJECT, not the GPU

MEASURED on n310 (A100), identical scene/params:
- forward: **3,100.3 s**
- width adjoint via **standard import source**: **3,133.6 s (52 min) — RAN ON GPU**
- same adjoint via lumopt2's **FieldRegion** object: CUDA `invalid
  configuration argument`, dies in seconds (3 independent tasks)
- same adjoint on **CPU**: 31,259 s / 43,538 s (8.7-12.1 h)

⇒ **The GPU engine CAN inject an arbitrary weighted field sheet**; only
lumopt2's (new, GPU-undocumented) FieldRegion source is rejected. Switching
the injection to `addimportedsource` (spec.wg_source="import") buys a
**10-14x** speedup and makes the EXACT FWHM gradient viable in-loop:
per-iteration ~2.6 h vs ~1.7 h (+50%, not +500%).
★Normalization differences between injection objects are absorbed by
C_field — that is exactly what the C-recipe calibrates.
**MY ERROR, cheap:** task 19 was given the 2 h lane; forward+adjoint+
contraction needs >2 h, so it TIMED OUT during contraction and never printed
its vector. Lesson: size the lane for fwd+adj+contraction (>=3 h), and for a
gradient GATE use the 12 h lane. Task 19 not re-run — its question is
answered; job **136122** (task 20, validate_gradient, 3 classes, GPU import
source, 12 h lane) now produces FD + adjoint for the C_field fit.

## 15. CHECKPOINT 2026-08-22 ~21:00 — live jobs + what each decides

| job | what | decides |
|---|---|---|
| **136107** | v2proj campaign (projection arch, re-trimmed-best seed) | does the optimizer improve on T 0.9597 at in-band width? first eval contracting |
| **136122** | task 20: FD gate on the **GPU** width adjoint (3 classes, 12 h lane) | C_field ⇒ unlocks the EXACT in-loop FWHM gradient |
| **136118** | decomposition ladder (depth / +cavity / +shifts) | is the +0.066 gain "just a deeper grating"? |
| staged | `campaign_v2_seesaw` (2nd basin, exact width gradient, GPU import source) | dispatch when C_field is known |
| held | MX-GRAD re-run (4 indices, after quota headroom) | can campaigns move to the production mesh convention |

PARKED FOR USER: (1) restart 136107 to pick up the four recovery-path fixes
(§13) — it is early, cost ~2 h, needs scancel approval; (2) production
confirm of any winner (N≈165 conformal accurate + lock-target re-trim + comb
δx re-check); (3) git commits (large uncommitted inventory).
HEADLINE RESULTS SO FAR (all MEASURED, surrogate level): re-trim +0.0665 T at
equal width (Q_i 36k→~103k); ρ-neutral shape +0.0317 T at +1.83%; both
survive the corrected pitch-locked mesh; lumopt2 ≡ regular physics to 18 pm.

## 16. FWHM-GRADIENT DERIVATION AUDIT (user request, 2026-08-22 late)

Full-chain re-derivation vs the implementation, line by line:
I(x)=Σ_c∫|E_c|²w_y dy → softW[I] (=envelope-FWHM as ε→0, co-area identity) →
∂softW/∂E_c* = (dsoftW/dI)·w_y·E_c → adjoint source conj(∂F/∂E*) =
W·conj(E_fwd), W=(dsoftW/dI)·w_y real → engine code IDENTICAL ✓. Jacobian
bookkeeping: fct sees softW as a value; its jac slot multiplies the field
contraction that carries dsoftW/dε — one slot, no double counting ✓. Crop,
trapz weights, single-λ plane, z-broadcast all consistent with profile_line ✓.
TWO NUANCES, gate-arbitrated by design: (a) import-source normalization ≠
dipole-sheet scaling — a single complex constant at one λ = exactly what
C_field fits; (b) one-way injection — the source plane sits ON the z-symmetry
boundary of a z-symmetric scene (BC mirrors it). If either argument is wrong,
the FD gate (136122) CANNOT pass a single-C fit — the gate is the proof.
ACTION under user authorization ("stop runs that run incorrectly"):
**136107 cancelled and redispatched as 136141** on the fixed engine — a
multi-day campaign with a broken cold-restart path on a preempting cluster is
incorrect by the program's own standard (§13 bugs A-D now in effect; log
resumes what existed).
- **Source-normalization audit COMPLETE (user concern, 2026-08-22 —
  MEASURED from the built scene):** monitor/twin/source all at z=0 = the
  mirror plane (validated empirically by the 18 pm identity test); BC is
  Anti-Symmetric (PEC-like) — the TM field at z=0 is E_z-dominant (tangential
  ~0 by parity, matching the BC), and a normal-E sheet on a PEC-like plane
  images to a CONSTANT ×2 — absorbed by C_field. Source defaults: Forward /
  amp 1 / phase 0 / override=1. Known import-source π/2 phase quirk (Ansys
  forum) = constant, absorbed by C_field's phase — this class of quirk is WHY
  the C-recipe fits a complex C instead of assuming (1,0). GAP FIXED: source
  wavelengths now pinned to the twin λ (136122 predates the fix — a near-zero
  adjoint there = off-λ spectrum, rerun after; a finite one = fine).
  What C_field CANNOT absorb (the honest list): spatially-varying source
  distortion. Candidates ruled out: dataset↔source grid mismatch (same
  monitor grid), tangential-E clash with the BC (parity-zero). Remaining
  candidate: the directional plane-wave decomposition of Forward injection —
  arbitrated by the FD gate's single-C fit across classes, per design.
- **DECOMPOSITION rung 0 (MEASURED, 136118_0): depth alone is WORSE than the
  origin** — flat corr 368.5: T 0.8573 at 15.909 um; normalised to the origin
  width: **0.8532 vs origin 0.8905 (−0.037)**. ⇒ "just a deeper grating" is
  REFUTED: uniform deepening LOSES transmission at equal width. The +0.066
  gain lives in shape/cavity/shifts — rungs 1-2 (running) apportion it.
