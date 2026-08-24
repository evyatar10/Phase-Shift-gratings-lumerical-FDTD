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

## 17. DECOMPOSITION COMPLETE (job 136118, all 3 rungs MEASURED 2026-08-23)
## — the gain is SHIFTS + CAVITY, NOT depth and barely shape

★SIGN CORRECTION first (my error, found while reading the rungs): the runner
normalised with `T + rate*(fw − 17.7136)`, which PENALISES a narrow rung.
Along the depth axis T and width move TOGETHER (deeper ⇒ narrower ⇒ lower T,
retrim curve 136051), so a narrow rung must be CREDITED the T it would gain on
being let back out to the origin width: `T + rate*(17.7136 − fw)`. Fixed in
retrim_decompose.py:69. The correction is ±0.004 and changes NO ordering.

Read from `results/retrim_decompose/results/retrim_decompose/rtdec_*_evals.jsonl`
(MEASURED, identical numerics: 50 nm PVA, v2 ±5 nm/501 window, N=100 surrogate):

| rung | T_pk | FWHM µm | Q_i | loss | T at origin width | Δ vs previous |
|---|---|---|---|---|---|---|
| origin (uniform+comb) | 0.89052 | 17.7136 | — | — | 0.8905 | — |
| a. depth only (corr 368.5 flat) | 0.85726 | 15.909 | 28 288 | 0.1421 | 0.8613 | **−0.0292** |
| b. + cavity 800→960.9 | 0.89819 | 15.940 | 41 125 | 0.1012 | 0.9022 | **+0.0409** |
| c. + tooth shifts (2Σs 130.6) | 0.95519 | 17.737 | 94 480 | 0.0435 | 0.9551 | **+0.0529** |
| d. + apodisation SHAPE (= full retrim) | 0.95968 | 17.755 | ~103 k | ~0.038 | 0.9596 | **+0.0045** |

VERDICTS (each MEASURED, ordering robust to the payback rate — the width
corrections are ≤0.004 against increments of 0.03-0.05):
1. **"Just a deeper grating" is REFUTED** — uniform depth alone LOSES 0.029 at
   equal width. Depth is the width knob, not the T knob.
2. **Cavity width is a nearly WIDTH-NEUTRAL T lever**: +0.041 T for +0.03 µm
   width (0.2%). In a width-constrained problem that is the cheapest gain in
   the inventory — and it is a SINGLE parameter (I_CAV). Campaign implication:
   keep wcav free with its full trust radius (currently 12 nm — it moved 160 nm
   in the best design, so the campaign can only creep there; flagged in §12).
3. **Tooth shifts are the dominant lever AND the width restorer**: +0.053 T
   while carrying the mode 15.94 → 17.74 µm back onto spec. This is direct
   evidence for keeping the 25 free shifts in the parametrisation (they were
   the parameter class the σ-era campaign under-used).
4. **Apodisation shape contributes only +0.0045** (6.5% of the +0.069 total).
   The corrugation PROFILE is nearly spent; a flat-corr device with the right
   cavity and shifts already captures 93% of the gain.
5. Caveat (honest): a one-path ladder attributes interactions to whichever
   ingredient is added later. The ordering was chosen depth→cavity→shifts
   because that is the order of increasing parameter count; a different order
   would move credit between (b) and (c), not between them and (a)/(d).

## 18. FD GATE 136122 — DIED OOM. Root cause + the fix (2026-08-23)

STATE: `OUT_OF_MEMORY`, exit 137, killed at 01:58:46 in "Computing gradient
fields from forward + adjoint data" — the SAME stage where the previous
attempt (136108) hit the walltime. Two attempts, ~3.7 GPU-h, zero physics
learned beyond what 136108 already gave (GPU adjoint = 3,135 s ✓).

ROOT CAUSE (read from lumopt2 source, zero GPU): `base_fom.calculate_
gradient_fields` (base_fom.py:473-511) does phase 1 = collect `e_fwd` for
EVERY fom entry into a list, phase 2 = load each entry's `e_adj` and
accumulate. Each array is (nx,ny,nz,3) × n_wl complex128 over the whole
optimization region ≈ 50 MB per λ here, so ~25 GB per full-grid array. A
MixedFom (port + width) therefore holds a THIRD cached region read — the
width adjoint's own file, whose region monitor records the full 501-λ grid
even though the FOM uses ONE λ — on top of the port-only campaign's peak,
which already fits 160 G only with modest headroom (136141: contraction
277.7 s, survives). +25 GB ⇒ kill.
Contributing (mine): the lane was sized 2 h for a job that needs
fwd + adj + 2 forwards PER INDEX = ~6 h. BOTH failed attempts were sizing
errors, not physics errors.

FIX (applied, offline):
1. `_w_spec` no longer hardcodes n_wl_points; task 20 runs at **151**
   (66 pm, odd ⇒ the centre λ stays on-grid, ~11 points across the 0.73 nm
   peak so the logged diagnostics stay meaningful). For a `wg_pure` gate
   (J = −softW) the T spectrum enters the FOM NOWHERE — softW comes from
   the single-λ twin at the scan centre — so this is the SAME physics at
   ~1/3 the contraction memory and cost.
2. Re-dispatch lane written into the runner:
   `SBATCH_MEM=250G LUMOPT2_QOS=12h_4g LUMOPT2_TIME=09:00:00`.
3. ★PRE-EMPTIVE: `campaign_v2_seesaw` (the exact in-loop width gradient)
   has the same MixedFom memory profile AND needs the full 501-point window
   for the T FOM — it would OOM at 160 G on iteration 0. Its docstring now
   dispatches at 300 G. This bug would have burned a multi-day campaign.

NOT AFFECTED: campaign 136141 (v2proj) — port-only FOM, one entry, measured
fine at 160 G.

## 19. IS THE EXACT FWHM GRADIENT WORTH IT? — decision rule, not a guess
## (user's own question: "let me know if this is actually helpful")

The FD gate is NOT on the critical path of anything currently running, and
after two failed attempts it is worth asking what it buys BEFORE spending a
third ~6 GPU-h. The honest accounting, with today's evidence:

AGAINST spending it now:
- The projection wall already holds: v2proj iteration 0 sits at FWHM 17.7532
  µm = **ratio 1.0022** of the anchor, mid-band (MEASURED, 136141).
- §17's decomposition puts the ENTIRE gain on tooth SHIFTS (+0.053) and
  CAVITY WIDTH (+0.041) — both differentiated by the ALREADY-VALIDATED port
  adjoint (C = 1.0561+0.1239i). The width gradient refines a CONSTRAINT, not
  the objective.
- Cost is not one-off: exact mode adds a second adjoint ⇒ ~+50%/iteration for
  the whole campaign, plus 300 G memory (§18).
FOR spending it:
- The wall's slopes (FW_A_MCORR, FW_A_ELONG) are LOCAL linearisations; far
  from the anchor they go stale, and the guard then acts as a fence
  (rejections) instead of a gradient (steering).

★DECISION RULE (apply to 136141's own log — free, no new runs):
- After ~10 accepted iterations, if fwhm_env/17.7136 has stayed inside
  [0.98, 1.02] AND WidthTrip has fired ≤1 time ⇒ the projection wall is
  ADEQUATE: skip the gate, keep both basins on the projection architecture,
  bank the ~6 GPU-h and the +50%/iteration.
- If the ratio walks to a band edge, or WidthTrip fires repeatedly (the wall
  is mis-centred and the optimizer is being fenced rather than steered) ⇒
  re-dispatch the gate at the §18 lane and go exact.
SHIPPED so this is a one-line switch either way: campaign_v2_seesaw.py now
carries `EXACT_WIDTH_GRAD` (default False = projection, dispatchable with no
gate at 160 G; True = exact, needs the fit and 300 G). The second basin is
therefore no longer BLOCKED on the gate — only its architecture is.

PARKED FOR THE USER (needs a go, it is multi-day GPU): dispatching the
see-saw second basin in projection mode. Recommendation: yes, once 136141
has shown ~5 healthy accepted iterations — two independent basins is this
programme's own convergence-evidence standard, and the see-saw seed carries
the measured −31% loss lever a smooth seed cannot discover.

## 20. ★T REPEATABILITY FLOOR ≈ 0.002 AT THESE NUMERICS (MEASURED 2026-08-23)

The two iteration-0 rows of 136141 are the same design evaluated twice.
Diffed server-side: **0 of 191 parameters differ**. Yet:

    lam 1566.064100  T 0.960179  fwhm_nm 0.727421   (row 1)
    lam 1566.064036  T 0.958170  fwhm_nm 0.727622   (row 2)

λ agrees to **0.064 pm** and the linewidth to 0.03 %, so the resonance did
NOT move — but peak T differs by **0.00201**. Sub-grid peak sampling is
therefore REFUTED as the cause (that would have moved λ_pk).
INFERRED mechanism: auto-shutoff truncation. Row 2 is simultaneously LOWER
in peak T and BROADER in linewidth — the exact signature of a DFT integral
truncated a little earlier in the ringdown (photon lifetime here ≈ 1.8 ps).
Production shutoff is 1e-7 for all runs (settled), so this is that setting's
floor, not a bug.

★CONSEQUENCES — apply to every campaign verdict:
1. **T differences below ~0.002 are NOT resolvable.** Report a T gain only
   above ~0.004 (2× the floor); anything smaller is "at the floor".
2. Q_i inherits it amplified: ±0.002 in T is ±5 000 in Q_i near T→1. Quote
   Q_i as ~10^5, never 5 digits.
3. The FOM gap between these two identical rows is 0.001 — i.e. L-BFGS-B can
   accept or reject a small step on noise. Watch for an accepted-best chain
   whose FOM gains are all ≲0.002: that is a noise walk, not convergence.
4. n = 2. This is ONE pair; if it starts mattering for a verdict, measure the
   floor properly (repeat a row 3-4 times) rather than leaning on this.

## 21. ★NIGHT-RUN PLAN 2026-08-23 (set on Fable; EXECUTE AS WRITTEN on Opus)

FLEET (all Athena, dispatched + verified ~00:15):
| job | what | lane | state at dispatch |
|---|---|---|---|
| 136141_0 | basin 1: v2proj campaign (retrimmed-best seed) | 4d_1g/160G | RUNNING, eval 3, in-band |
| 136188_0 | basin 2: v2uniform campaign (STRICTLY uniform seed — user decision; NO trust box, seedA budgets 60/100) | 4d_1g/160G | RUNNING, clean startup |
| 136189_20 | W3-GPU FD gate, FIXED (151 pts, import/GPU) | 12h_4g/250G/9h | RUNNING |
| 136190_21 | Im-quadrature partner (adj_fix_field=(0,1), same detune-1 point/indices/config) | 12h_4g/250G/4h | PENDING afterok:136189 |
Monitor b00yes3jh: 15-min poll, one ssh, change-key only (states, eval
counts+last row both campaigns, gate marks, err-files, quota).
License at dispatch: 10/50 used. Quota 248G/300G, cleaner v2 running.

RULES FOR EVERY WAKE (no re-derivation — apply, log, move on):
1. CAMPAIGN HEALTH (each basin separately): accepted-row test = T rising
   or fwhm falling; band = fwhm_env/17.7136 ∈ [0.95, 1.02] (basin 1's
   anchor updates on re-anchor; read wg/fw fields from the row).
   T differences <0.004 are NOISE (§20 floor 0.002) — never call a gain
   on less, and a chain of ≲0.002 "gains" = noise walk ⇒ note it, don't
   celebrate it.
2. GATE 136189 done ⇒ pull the printed (fd, adjoint) vectors from
   lum_array-136189_20.out; 136190 fires automatically (afterok) and adds
   the Im vectors in lum_array-136190_21.out. When BOTH exist: paste the
   three vectors into fit_c_field.py (local), PASS = sign 3/3 + single-C
   residual ≤10% every class. Record C_field in §22 + skill. DO NOT flip
   any campaign to exact mode mid-run — EXACT_WIDTH_GRAD is for a FUTURE
   restart, decided by §19's rule (10 accepted iters, WidthTrip count).
   If 136189 OOMs/timeouts AGAIN at 151/250G/9h: STOP, do not resize and
   retry — that's a third sizing failure = something else is wrong; hold
   for free diagnostics (MaxRSS via sacct, log stage) + user.
3. FAILURE SIGNATURES: <30s crash = stale code/license (check "Simulation
   time" ~1s ⇒ license no-op); oom_kill in log = memory (record MaxRSS);
   REQUEUE of a campaign = benign (resume ≤1 eval loss) — just verify the
   restart row matches the pre-kill best. Job vanished from squeue but
   sacct COMPLETED + campaign not converged ⇒ walltime — re-dispatch same
   module (resume).
4. QUOTA: WARN ≥270G — verify cleaner alive (`pgrep -f h5_roll_clean`),
   worst case ask user before any deletion beyond the cleaner's rules.
5. CONNECTION BUDGET: monitor's 1 ssh/15min is the whole automated budget;
   wakes answer from the event line + at most ONE extra ssh.
6. MORNING REPORT: per basin — accepted iters, best (T, fwhm ratio, λ),
   WidthTrip count, §19 verdict progress; gate verdict + C_field if fitted;
   any failures + what was done. Label MEASURED/DERIVED/EXPECTED.
PARKED (user-only): flipping exact mode on; production confirm; MX-GRAD
rerun; git commits; any deletion.

## 22. ON-RESONANCE SAMPLING AUDIT (user question 2026-08-23, pre-Opus-switch)

Three consumers of the field profile, three different answers (all from code):
1. **fwhm_env (the width AUTHORITY in both running campaigns): ON-RESONANCE
   by construction** — extracted at the FOUND lam_pk (engine line 1178,
   profile_line(fdtd, lam_pk)) from the 501-pt/20-pm grid ⇒ ≤10 pm =
   0.014 linewidths off. NOT undersampled: 36 samples across the 0.73 nm
   line; λ_pk repeatability measured at 0.064 pm (§20).
2. **The softW twin (FOM carrier in width_grad mode): pinned at SCAN CENTER**
   (engine line 862) — NOT at the per-eval resonance. Irrelevant to the
   running campaigns (they are projection-mode; nothing reads the twin).
   At W2 the offset was 64 pm = 0.09 linewidths (fine). At the GATE's
   detune-1 point the offset is UNKNOWN → task 22 (job 136198, chained
   afterok:136190) measures it: offset ≲1 linewidth ⇒ C fit transfers;
   ≫1 linewidth ⇒ re-gate at a corrected center before ANY exact-mode
   campaign. THIS IS THE §21 rule-2 amendment: judge the C fit ONLY
   together with 136198's offset.
3. **Exact-mode campaigns (future): known limitation** — the twin follows
   scan_center, which recenters only on restart; designs drift ~1 nm
   between recenters ⇒ twin up to ~1.4 linewidths off-peak. If exact mode
   is ever adopted: pin the twin per-eval to the previous eval's lam_pk
   (engine change + re-gate) or tighten the recenter threshold. PARKED.
λ-grid adequacy: campaigns 501/20 pm (peak-T sampling error ≪ the 0.002
shutoff floor); gate 151/66 pm justified because its FOM never reads T(λ).
Deploy verification: local md5 == remote md5 for campaign_v2_uniform.py
and validate_c325.py (checked post-dispatch).
CHAIN NOW: 136189 (gate) → 136190 (Im-quad) → 136198 (detune-point
resonance offset). If 136189 fails, release with
`scontrol update job <id> dependency=''` — else the chain pends forever.

## 23. USER RULING 2026-08-23 ~01:00 + actions (supersedes parts of §21/§22)

★USER RULING: softW at scan center is UNACCEPTABLE — softW must be measured
ON RESONANCE, always. Standing requirement for anything width-gradient.
- Campaigns (136141/136188): COMPLIANT as running — code-verified, every
  logged softw_um comes from the lam_pk profile (engine :1197); the
  center-pinned twin is read only under width_grad=True. NO rerun needed.
- Gate chain: order was WRONG (my mistake — the offset measurement was
  chained LAST). FIXED: 136198 dependency RELEASED, runs NOW (~1.2 h).
  RULE: |lam_pk − 1564.21| ≤ 0.73 nm (1 linewidth) ⇒ gate 136189/136190
  stands; > 1 linewidth ⇒ cancel remaining chain (scancel prompts), set
  scan_center_nm=<measured> in tasks 20/21 (_w_spec kwarg), re-dispatch
  gate+quad (§18 lanes) — still done by morning. If lam_pk sits AT the
  window edge (±5), the resonance is outside — recenter wider and re-probe
  before re-gating.
- EXACT-MODE PRECONDITION (hard, from the ruling): twin must track the
  per-eval resonance (engine change + re-gate) BEFORE any exact-mode
  campaign. EXACT_WIDTH_GRAD stays False until then.
QUOTA incident: h5_roll_clean was DEAD (nothing running); restarted
(PID 2113402). Cleaner-immune scratch (newest-2-per-dir in one-shot gate
dirs): 13 stale *_output.h5 >3 h deleted WITH user approval → 243→206 G.
Night worst-case +40-50 G transient ⇒ fits. Monitor wakes on quota change;
270 G = investigate, 300 G = soft limit, 330 G = hard kill line.

## 24. ★★★MESH-PHASE WIDTH ARTIFACT — MEASURED, ROOT-CAUSED, CAMPAIGNS
## MIGRATED TO THE PITCH-LOCKED MESH (2026-08-23 night, autonomous)

THE FINDING (all MEASURED locally on real+synthetic profiles, zero GPU):
fwhm_env is built through STANDING-WAVE PEAKS; the wave has the pitch
period (516.6 nm measured on ev0001). At dx=50.0 nm (10.34 samples/period,
non-integer) the sampling phase drifts across the mode and fwhm_env
mis-reads by up to **700 nm (3.9%) on the real profile** and −1046 nm worst
on a known-truth synthetic, purely as a function of where the wave sits on
the grid. Tooth shifts TRANSLATE the wave ⇒ the error is DESIGN-DEPENDENT
and does NOT cancel in ratios — it is the mechanism behind §11's measured
+3.6/+0.6/+2.6% spread. Against a +2/−5% band this eats most of the
tolerance. At dx=pitch/10 (exactly 10.00/period): synthetic error +0.1 nm,
phase-INDEPENDENT; real-profile spread 0.56%. softW is immune either way
(≤3 nm) — smoothing kills the phase sensitivity. Also measured: both width
operators are smooth (0 non-monotonic steps under stretch; zero 2nd-diff
kinks along a real I1→I2 path), amplitude-invariant, and the profile save
grid is uniform 50 nm over ±51.66 µm.
ACTION (user mandate "must not undersample; fix in advance"): both
campaigns migrated to region_dx_nm = eng.DX_PITCHLOCK_NM (= PITCH_NM /
CELLS_PER_PITCH, DERIVED per user order — never hardcode the quotient).
Labels *_px; anchors = MEASURED mx rows (§11): uniform fwhm0 18.3460 /
softw 18.476441 / λ 1564.614; retrim-seed fw 17.8530 / λ 1566.377 (ratio
0.9731, in-band). 50-nm campaigns 136141/136188 CANCELLED (~18 GPU-h,
evals unusable for width decisions — the width channel was the artifact);
corrected campaigns: **136248 (basin 1) / 136249 (basin 2)**, both RUNNING
on 4d walltimes.
GATE CHAIN DISPOSITION: 136198 measured the detuned twin offset = +0.724 nm
= 0.99 linewidths (≈33% resonant fraction) — borderline-keep per §23, so
136189/136190 finish as a MECHANISM gate (sign structure + single-C-fits-
all-classes), but their C_field is at the 50-nm mesh and off-peak ⇒ NOT for
production exact mode. Exact mode now requires ONE consolidated re-gate:
pitch-locked mesh + twin-tracking fix + on-resonance center. (136189 also
ate two contrib preemptions, restarts from zero — 3rd attempt running.)
GATE 136189 COMPLETED 3:37:33 (attempt 3). MEASURED vectors at detune-1,
indices [corr_0, shift_0, wcav], J=−softW, naive C=(1,0):
  FD  = [-0.00365, +0.01825, +0.02026]
  Re  = [ 0.0000 , -0.0352 ,  0.0000 ]   (err 208%)
Reading: EXACTLY the known quadrature pattern (skill item 6) — corr/wcav
are λ-shifting ⇒ arg Z ≈ 90° ⇒ Re{Z} ≈ 0, shift's Re wrong-sign. NOT a
fail signature by itself. VERDICT WAITS for 136190's Im vectors (running):
finite Im for corr/wcav + single-C fit ≤10%/class = mechanism PASS (at
50 nm mesh, mechanism-only per §24); Im also ≈0 for corr/wcav = structural
zero ⇒ width-adjoint coupling broken for y-edge params — escalate.

## 25. FABLE REVIEW (2026-08-23 morning) — error hunt on the night edits +
## the twin-tracking fix (all LOCAL; nothing deployed, nothing dispatched)

### A. Errors found + FIXED (local edits, this session)

1. ★GATE PENALTY ASYMMETRY — contaminates tonight's C_field fit's shift row.
   lumopt2's `validate_gradient` (fd_grad.py:194, read this session) FDs
   through `project.fom.calculate_fom` = the RAW fct (NO attached penalty),
   but takes its adjoint from `project.compute_gradient` = the
   attach_penalty-WRAPPED one (raw − pen_grad). `run_adjoint_only` printed
   the wrapped gradient too. Invisible while pen_grad = 0 at the probe
   point; at the detune-1 gate point it is NOT zero: 25 shifts of 20 nm ⇒
   elong = 1000 nm > the 120 nm deadband ⇒ pen_grad = 2·1e-5·880·2 =
   **0.0352 on every shift param** (DERIVED, verified against
   `_kappa_penalty_grad` in the smoke) — the same order as d(softW)/dshift
   itself (~0.027). corr row clean (rho = 1.0 inside deadband), wcav clean.
   FIX (engine): attach_penalty now stashes `compute_fom_raw` /
   `compute_gradient_raw` (lumopt2_design.py:765);
   `run_validate_gradient` swaps in the raw gradient before calling
   lumopt2 (:1946) and `run_adjoint_only` prints the raw vector (:1980) —
   raw-vs-raw, matching what the FD side always measured.
   ★MORNING ACTION (136189/136190 ran the PRE-fix engine): fit_c_field.py
   now carries `PEN_GRAD = [0.0, 0.0352, 0.0]` (fit_c_field.py:36) and adds
   it back to RE/IM before fitting — use as-is for tonight's vectors; zero
   it for vectors printed by the fixed engine. Round-trip verified in the
   smoke (known C recovered to <1e-3 after correction).
   ★MEASURED CONFIRMATION (found in §24 after this analysis was complete —
   §24 landed in the file while this review ran): 136189's printed Re
   vector is [0.0000, −0.0352, 0.0000] — the shift row is EXACTLY −pen_grad
   to the printed precision. So §24's reading "shift's Re wrong-sign" is
   the penalty artifact, NOT physics: the true Re{Z_shift} ≈ 0, i.e. ALL
   THREE classes show the same clean quadrature pattern (arg Z ≈ 90°),
   which is the EXPECTED pre-C-fit signature, not a partial anomaly. After
   PEN_GRAD correction the fit reduces to the Im vectors (136190), as the
   quadrature recipe intends.
   COLLATERAL (flagged, not re-run): any post-elong-guard PORT-path
   validate/adjoint print at a detuned point has a swamped shift row
   (0.0352 vs fd ~4e-5 — ~880×) — MX-GRAD task 18, when re-run, is now
   clean; and before reusing the port C = 1.0561+0.1239i for any NEW fit,
   confirm its source vectors predate the elong guard (its 1.7%-on-7-params
   fit quality says they did — a contaminated shift row could not fit).
2. fit_c_field.py stale LABELS (5 old task-12 names vs the 3 indices tasks
   20/21 actually print) — zip() would have silently mislabeled shift_1 as
   "corr_25" and wcav as "avg_1". Fixed (fit_c_field.py:26) + length assert.
3. validate_c325.py task 22 printed `None` through a `+.3f` format — a
   failed resonance read would have raised TypeError and masked the actual
   result. Fixed (validate_c325.py:555); the print now also shows
   softw_adj_um (twin@center) next to broadband softw_um — the comparison
   the docstring describes. (Job 136198 was unaffected — it read a λ.)

### B. Checked and CORRECT (no change)

- campaign_v2_projection: scan_center 1566.377 / fwhm0 18.3460 (origin) /
  fw_anchor fwhm 17.8530 all ≡ the 136077 anchors ✓. Seed band ratio
  17.853/18.346 = 0.9731 ∈ [0.95, 1.02] ✓. seed_override ordering CORRECT:
  set BEFORE replay_params, so param_bounds' trust boxes AND replay's
  bounds-check both center on the retrimmed seed (replay against the
  uniform seed's sliver comb bounds would have crashed — the job-133395
  class). main()'s fw_anchor mcorr/elong = 368.47/130.62 ≡ the docstring's
  368.5/130.6 ✓ (smoke-verified); no corr touches the 500 cap (max 374.8).
  Trust box holds the seed on every param ✓. Note (deliberate, not a bug):
  trust radii shrink at physical edges, so the smallest shift teeth
  (0.88-6.3 nm) get ±0.88..±6.3 boxes, and per-restart re-centering is the
  escape; wcav gets its full ±12 around 960.9.
- campaign_v2_uniform: scan_center 1564.614 / FWHM0 18.3460 / SOFTW0
  18.476441 ≡ mx_origin ✓ (the uniform seed IS that scene — free_comb only
  changes addressing, not the build). fw_anchor {18.3460, 325, 0} exact for
  this seed ✓. EXACT_WIDTH_GRAD=False path touches nothing width_grad
  (width_grad stays False, no twin is built, attach_penalty takes the
  fwhm_wall branch) ✓. No trust box ✓.
- validate_c325 tasks 20/21: SAME operating point (both detune=1 — same
  four p-edits in run_validate_gradient and run_adjoint_only) and SAME
  config (wg_pure/import/GPU/151 pts; only adj_fix_field differs, as
  intended) ⇒ the C fit is valid once PEN_GRAD-corrected. n_wl_points=151
  reaches the spec (_w_spec passes it through dataclasses.replace via kw)
  ✓. Task 22 arithmetic ✓: measured offset 1564.934−1564.21 = +0.724 nm =
  0.99 linewidths — PASSES the §23 ≤1-linewidth rule by 6 pm only;
  marginal, but the rule as written stands (and the fit's own residual
  gate remains the arbiter).
- §13 recovery fixes A-D all present in the engine: A cold-restart
  fw_anchor re-anchor (run_campaign fwhm_wall branch), B no corr-cap-down
  under a width wall (WidthTrip handler), C fail-closed width filter
  (_best_from_log), D loud fwhm_wall-without-anchor assert
  (attach_penalty :707).

### C. FW_A_MCORR / FW_A_ELONG at the pitch-locked mesh — VERDICT: KEEP, safe

The slopes (engine :554-555) were fitted on 50-nm-mesh rows whose LEVELS
carry the ±3.9%-class phase error, and the error is design-dependent, so the
slopes themselves may be off by tens of percent (worst case ~±0.027 µm/nm on
−0.047 across the 52.5 nm retrim span). That is acceptable BY CONSTRUCTION:
the wall is a steering hinge between measurements, not an authority — the
anchor is re-measured (pitch-locked fwhm_env, 0.56% class) at every accepted
best, so the slope only extrapolates over one accepted step. v2proj: step ≤
trust 12 nm ⇒ worst predicted-fhat error ~0.3 µm vs a 0.64 µm half-band —
worst case is a hinge that engages early/late by a fraction of the band,
never a delivered violation (measured guard + fail-closed filter own that).
v2uniform (no trust box): a large early step can outrun the linearization,
but that is the SAME already-accepted risk as extrapolating from mcorr
368→325, and §19's decision rule (WidthTrip count / band-edge walk) is the
standing monitor. Do NOT spend solves remeasuring slopes now; refit only if
§19's rule trips.

### D. Twin-tracking fix — flag `wg_track_resonance` (default OFF, inert)

Scope of the bug (verified in code): ACROSS restarts the twin already
follows — run_campaign recenters scan_center_nm on the best row's λ and
make_project→build_base_fsp rebuilds the twin there. The gap is WITHIN a
segment: recenter trips only on an accepted best drifting >RECENTER_NM =
2.0 nm ⇒ the twin can be 2.7 linewidths off before a restart, and at the
measured ~1 nm/accepted-iteration drift it is ~1.4 linewidths off after ONE
accepted step — far outside the ~0.3-linewidth (0.22 nm) requirement.
MECHANISM (smallest that survives regeneration): with the flag ON, the
parametrization func emits ONE extra prop,
`field_profile_adj::wavelength center` = spec._wg_lam_track (the log
callback advances it to lam_pk on EVERY eval, probes included; scan center
until the first eval). Props go through lumopt2's documented
`_update_properties` channel (parametrization.py:610 — read this session),
re-applied at every project (re)generation, unlike setnamed-after-generate()
which is wiped. The value is a constant to autograd (zero Jacobian row, zero
dEps). The import-source λ-pin follows FOR FREE: setup_adjoint_simulation
pins width_adj_src to sim_result.wavelengths[0], which WidthResults.
get_results sets from the twin's own RECORDED λ. Residual offset = one
eval's λ drift ≈ ≤0.3 linewidths at accepted cadence; a large line-search
probe (measured up to +2.6 nm) mis-samples only its own eval and
self-corrects on the next; same-eval exactness would need a second forward
(λ_pk is unknown until the forward ran) — rejected as not minimal.
Engine edits: CampaignSpec :263-282, make_func :433-434/:466-474, callback
:1222-1226. SMOKE (local, zero lumapi): compileall clean; both campaign
SPECs unaffected (flag False, no twin key in func props); flag ON emits the
key at scan center and follows _wg_lam_track. §23's exact-mode precondition
is met by an ENGINE change + re-gate: run W2/W3 with the flag ON before any
exact-mode campaign — the flag stays OFF everywhere until that gate passes.

### E. Deliberately NOT changed

- The running campaigns 136248/136249 and their remote code: everything
  here is local; validate_c325.py/fit_c_field.py/engine now DIFFER from the
  deployed copies — re-deploy (--upload-only) before the NEXT gate
  dispatch, never mid-flight.
- fwhm_wall re-anchoring on every ACCEPTED BEST inside a segment makes the
  objective mildly non-stationary under L-BFGS-B. Deliberate item-24 design
  (anchor tracks truth); noted, not touched.
- v2proj seed sits 10.6 nm outside the 120 nm elong deadband (2Σs = 130.6)
  ⇒ a standing elong_penalty tax of ~0.001 FOM and 4.2e-4/shift gradient.
  Pre-existing and intended (the cheat wall is absolute, not seed-relative).
- The ≤1-linewidth ruling on the 0.99-linewidth gate offset (§23): user's
  rule as written; the marginal pass is recorded in B, not re-adjudicated.
- RECENTER_NM=2.0 and the recenter-on-accepted-best-only policy: correct
  for the T path; the width-side gap it leaves is what wg_track_resonance
  (OFF) now covers when needed.

## 26. PEAK-ENERGY vs FWHM AS THE CONSTRAINT — MEASURED VERDICT: NO SWITCH
## (user question 2026-08-23 morning; answered from 8 stored profiles, 0 GPU)

MEASURED across 8 real designs (profiles: v2proj seed, mx_origin/retrim/
rho15/ident, rtdec depth/cav/shift; FWHM span 15.91-19.25 µm):
- **Raw peak intensity I_pk vs FWHM: concordant pairs 3/21** — no monotone
  relation (weakly ANTI-monotone). Mechanism: I_pk ≈ U_stored/L_eff and
  U_stored moves with Q/loss across designs (sum_I varies 22% here), so the
  17.75-µm best design peaks HIGHER than the 15.91-µm depth-only one.
  A peak constraint would reward energy storage, not width — the optimizer
  would raise Q_L against it. INVALID as a width surrogate.
- **Normalized peak U/I_pk (participation length)**: monotone with FWHM on
  these 8 — BUT this is exactly the participation-ratio family already
  measured to err 21 pp as an fwhm tracker where softW errs ≤2.2 pp (§2),
  and it is shape-blind the same way σ is: for a pure exponential envelope
  FWHM = ln2·L makes ALL these measures proportional — they diverge exactly
  when apodization reshapes the envelope, which is the regime the program
  optimizes in. Literature check: peak-normalized quantities are the
  standard "mode volume" objective in cavity inverse design (Q/V work),
  which is a different physical target than an envelope FWHM overlap spec.
- **Smoothness motivation is MOOT**: §24 measured softW and fwhm_env both
  smooth (0 kinks) on real deformation paths; the gradient carrier is
  already softW (C∞ by construction).
DECISION: constraint stack unchanged — fwhm_env authority + softW carrier.
BONUS (definition stability, measured): FWHM threshold sensitivity is
dW/W ≈ 12.5-15.2% per 5% threshold shift on all three design types ≡ the
ANALYTIC value dt/(t·ln t)=14.4% for an exponential envelope. So "half" has
no special plateau — the definition is stable only because the threshold
inputs are: floor+peak repeat at the 0.2% class (§20) ⇒ width noise ~0.03%
through this channel; the px mesh fixed the peak-VALUE channel (§24). This
amplification is WHY the mesh-phase artifact was so large.
QUAD 136190 COMPLETED (1:40) — ★ESCALATION BRANCH FIRED: printed vector
IDENTICAL to the gate's ([0, -0.0352, 0]) and its header says
adj_phase_fix=False C=(1.0,0.0) — the (0,1) width-C rotation apparently
never engaged. After PEN_GRAD subtraction BOTH raw adjoint vectors are
[0,0,0]: the width-adjoint contributes (print-precision) ZERO to every
parameter — not quadrature, a structural/plumbing break. Per §6: NO new
gate dispatches until root-caused (Fable agent on it; server h5 forensics
next). Projection campaigns 136248/136249 are width_grad=False and believed
unaffected (verification requested from the review agent).

## 27. ★★★WIDTH-ADJOINT ZERO-GRADIENT ROOT CAUSE (Fable review 2, 2026-08-23)
## — the GPU import source injects NOTHING from the z=0 plane

MEASURED inputs: 136189 FD = [−0.00365, +0.01825, +0.02026], wrapped adjoint
= [0.0000, −0.0352, 0.0000]; 136190 (C_field=(0,1)) printed the IDENTICAL
vector. After the §25 PEN_GRAD subtraction both raw adjoint vectors are
[0, 0, 0] to print precision — the width entry contributed nothing, and
rotating nothing by i changes nothing (hence identical prints).

### Verdict: (B) physics/pipeline — with one cosmetic (A) print bug

(A) REFUTED on both sub-claims (lumopt2 source read locally, base_fom.py):
- The field C IS consumed: `_compute_adjoint_fields_phased` is a real
  base_fom hook (base_fom.py:366) called per entry in the accumulate loop
  (base_fom.py:497-511); MixedFom's override multiplies the width entry's
  fields by complex(adj_fix_field_re, adj_fix_field_im). The header line
  "adj_phase_fix=False C=(1.0,0.0)" printed the PORT knobs — cosmetic only.
  FIXED: run_adjoint_only now prints C_port AND C_field.
- The width entry is NOT dropped from the contraction: it sits in
  config_map with its own jac slot (fct = −x[n_wl]; autograd jacobian slot
  151 = −1 exactly; port slots 0), its own scaling factor (FieldFom's
  2·eps0·dV/base_amp — large, never zero), and its own e_fwd/e_adj reads
  from optimization_dft (base_fom.py:452-511 walked line by line). §18's
  OOM at this exact stage already proved the width entry's region reads
  happen.

(B) THE MECHANISM (each link labeled):
1. MEASURED (§16 audit): twin monitor + import source sit at z=0 = the
   ANTI-symmetric (PEC-like) z boundary, where tangential E ≡ 0 by parity —
   the TM field on that plane is Ez-dominant, tangential is numerical dust.
2. EXPECTED (documented Lumerical behavior; the diagnostic below makes it
   measured): a plane import source with injection axis z builds its
   equivalence currents from the TANGENTIAL field components only — the
   dataset's Ez does not inject. (The FieldRegion 'source mode' object is
   DIPOLE-based — z-polarized dipoles fine — which is why only the import
   route zeroes out; its scaling formula dipole_base_amplitude confirms the
   dipole model, field_fom.py:112/139.)
3. ⇒ the imported W·conj(E)|z=0 sheet is Ez-only ⇒ injected power ~
   (tangential/Ez)² of intended ⇒ e_adj ≈ noise ⇒ width gradient ≈ 0.
   Predicts tiny-but-not-machine-zero (print rounding hides the dust) —
   the h5 read below distinguishes this from an exact drop.
RETROACTIVE corrections this forces (honesty §9):
- §14's "★★★GPU WIDTH-ADJOINT PROVEN" is DOWNGRADED: 136108's 3,133.6 s
  import-source "adjoint" almost certainly solved an empty scene (runtime ≈
  forward's 3,100 s is consistent with a source-less run integrating the
  full sim time). "Runs on GPU" is proven; "computes the width adjoint" is
  NOT. The 10-14× speedup claim is void until a source that demonstrably
  injects passes the FD gate.
- The CPU FieldRegion timings (8.7-12.1 h) were real adjoints (dipole
  injection unaffected) — the cost verdict stands.
- 136189's FD half is genuine, first-of-its-kind data: MEASURED
  d(softW)/dp at detune-1 = [−0.00365 (corr_1), +0.01825 (shift_1),
  +0.02026 (wcav)] — keep for any future gate (FD is config-independent).
  Note softW RISES with shift_1 and wcav at this off-peak sampling point.

### Server-side free diagnostics (existing files, login-node python3, ONE ssh)

Base: ~/bragg_sim_athena/results/validate_c325/results/lumopt2_val_c325/
lumopt2_val_c325_w3gpufd_files/ (and the _w3gpuquad_files sibling). .fsp and
*_output.h5 are HDF5 — internal paths undocumented, so the script walks and
classifies. Decisive numbers: (a) max|E| per component of the twin monitor
in the FORWARD file — expect |Ex|,|Ey| ≪ |Ez| (mechanism premise, makes
link 1 quantitative); (b) max|E| of the optimization_dft data in the
ADJOINT file vs the FORWARD file — ratio ≲1e-4 ⇒ dead source CONFIRMED;
ratio O(1) ⇒ contraction-side drop instead (revisit). Script:

    python3 - <<'PY'
    import h5py, numpy as np, glob, os
    base = os.path.expanduser('~/bragg_sim_athena/results/validate_c325/'
        'results/lumopt2_val_c325/lumopt2_val_c325_w3gpufd_files')
    def cabs(a):
        a = np.asarray(a)
        if a.dtype.names and {'r','i'} <= set(a.dtype.names):
            return np.abs(a['r'] + 1j*a['i'])
        return np.abs(a)
    def scan(path, tag):
        fs = sorted(glob.glob(path))
        if not fs: print(tag, 'NO FILES', path); return
        f = fs[-1]; print('==', tag, f)
        hits = []
        def w(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.size > 300:
                hits.append((name, obj.shape))
        with h5py.File(f, 'r') as h:
            h.visititems(w)
            for name, shape in hits:
                low = name.lower()
                if not any(k in low for k in ('field_profile_adj',
                        'optimization', 'width_adj_src', '/e', 'ex', 'ey',
                        'ez')):
                    continue
                d = h[name][()]
                m = cabs(d)
                if shape[-1] == 3 and m.ndim >= 1:
                    print(f'  {name} {shape} max|Exyz| = '
                          f'{[float(m[...,c].max()) for c in range(3)]}')
                else:
                    print(f'  {name} {shape} max = {float(m.max()):.4e}')
    scan(base + '/fwd_*/*_output.h5', 'FWD')       # (a) + fwd region ref
    scan(base + '/adj_*field_profile_adj*/*_output.h5', 'ADJ-width')  # (b)
    PY

If the name filter prints nothing useful, drop the `if not any(...)` filter
to get the full tree (one file, prints are cheap). Also worth one glance in
the same ssh: the width-adjoint job log's autoshutoff column — a source-less
run never decays normally.

### Fixes applied (LOCAL, not deployed; smoke-verified)

1. `check_import_src_injects` guard (lumopt2_design.py, called in
   MixedFom.setup_adjoint_simulation's import branch before importdataset):
   RAISES when tangential source max < 1e-3 × Ez max — the silent 6-GPU-h
   zero becomes a 1-second loud failure. Unit-smoked: Ez-only plane raises,
   tangential-rich plane passes; compileall clean.
2. run_adjoint_only header now prints C_port and C_field (the misread that
   spawned hypothesis A).
3. NOT implemented (design work, user decision — §8 exploratory-question
   rule): resurrection routes for the exact width gradient, all gate-
   arbitrated: (i) twin+source on a z-offset plane (tangential E nonzero
   there; incomplete adjoint current — usable ONLY if the FD gate still
   fits a single C across classes); (ii) Ansys bug report: FieldRegion
   source GPU CUDA rejection (the dipole object is the CORRECT injector);
   (iii) stay projection-only (the running architecture; §19's evidence so
   far says the wall is adequate). EXACT_WIDTH_GRAD remains False and now
   ALSO blocked on a working injector, not just the C fit + twin tracking.

### Impact on running campaigns 136248/136249: NONE (verified, §26 request)

Both are width_grad=False ⇒ make_project takes the plain
`lmpt.Fom(port_res)` branch — MixedFom/WidthResults are never instantiated,
build_base_fsp never creates the twin or the import source, and the FOM/
gradient is the validated port path (C_port = 1.0561+0.1239i). Their width
numbers (fwhm_env/softw in the eval log) come from the BROADBAND
field_profile monitor read in the log callback — a monitor read, no adjoint
source involved. The zero-injection path is unreachable; the new guard
lives inside MixedFom only. The §25 review fixes are equally inert there
(raw-gradient swap touches only the two gate entry points; wg_track_
resonance defaults False).
### §27 FORENSIC RESULT (MEASURED, h5 read of 136189's solved files)
FWD twin plane (z=0): max|Ex|=0.0, |Ey|=0.0 EXACTLY, |Ez|=8.41 — the plane
is tangentially dead by parity, as §27 predicted. ADJ-width file: EVERY
monitor field EXACTLY 0.0 — the adjoint contained no field at all (dead
source at ratio 0, not 1e-4). Root cause closed end-to-end: import source
at z=0 cannot inject a TM (Ez-only) adjoint. All printed "adjoint" vectors
were the penalty gradient alone. GPU-import width-adjoint route is DEAD at
z=0; §14's speed claim void; FD half of 136189 = keep-forever data.

## 28. THREE-BASIN PROGRAM + THE SHIFT-NECESSITY QUESTION (user 2026-08-23)

FLEET (all Athena — see cluster rule below):
| job | basin | seed | shifts | answers |
|---|---|---|---|---|
| 136248 | 1 | retrimmed best | free (trust 12 nm) | can the good device improve? (preserves prior work) |
| 136249 | 2 | uniform origin | free, start 0, no trust box | what does a FRESH optimizer choose? (behaviour, not numbers) |
| 136297 | 3 | uniform origin | **FROZEN at 0** | the no-shift ceiling, by construction |
| 136296 | — | BEST_T9635 pre-retrim, px mesh | — | true retrim narrowing at an unbiased mesh |

★THE SHIFT TEST: basin 2 vs basin 3, both re-trimmed to EQUAL width, then
compare T. Basin 2 alone is ambiguous — shifts sit ON their lower bound (0),
so "they didn't move" could mean useless OR gradient pushing outward; basin 3
removes the ambiguity. Preliminary DERIVED estimate from the decomposition:
shifts worth ~+0.025 T at equal width (vs +0.057 raw) — to be replaced by
the measured basin2-vs-basin3 gap.
★RETRACTION (2026-08-23): the claim that a smooth seed "provably cannot
discover" the see-saw is WRONG — that direction has an ordinary nonzero
gradient. No see-saw seed dispatched; basin 2 can reach it if it pays.
★ALREADY-ANSWERED sub-question: "corrugation not locked around an average"
is ALREADY the parametrisation (per-tooth corr AND avg free, bounds
(150,500)/(775,825) — any narrow/wide pair reachable). The optimizer left
every avg within 1.9 nm of 800: the freedom exists and was declined.
★CLUSTER RULE (from tonight's MEASURED preemption record — 3 contrib
preemptions in 8 h; campaigns lost ≤1 eval each, the non-resumable gate was
destroyed TWICE, ~4 GPU-h): resume-protected CAMPAIGNS stay on Athena
(faster GPUs, cheap preemption, user preference); long NON-RESUMABLE
one-shots (gates, production confirms) go to IGUM (empirically calm) when
they exceed ~2 h. Seats are not binding (10/50 in use, ~2/campaign).
★LESSON (job 136283, 10 s death): a branch of main() used `dataclasses`
while OTHER branches import it locally ⇒ function-local name ⇒
UnboundLocalError. My module-scope smoke missed it. RULE: smoke a new task
THROUGH main(task_idx) with the solve stubbed, never through a
reconstruction of its body (same lesson as skill item 23).

## 29. ★SYMMETRIC ±2% BAND (user decision 2026-08-23) + campaign relaunch

USER RULING: the width band is now SYMMETRIC ±2%, was +2%/−5%. Rationale
(user, and correct): the acousto-optic spec is a FIXED interaction width —
a NARROWER mode buys nothing, so the −5% allowance let the optimizer spend
width it has no use for, and width is worth transmission (MEASURED payback
0.0030 T/µm). The ±2% keeps tolerance for measurement noise (§20 floor) and
removes the free narrowing. Engine: RHO_DN 0.95 → 0.98 (one constant; feeds
the fwhm_env authority, the fwhm_wall hinge and the retired rho penalty).
BENCHMARK CLARIFIED (this confused us for a while — record it):
| device | N=100 | ratio | production equiv (×19.91) |
| bare, no comb | 19.25 | 1.049 | 20.89 |
| **origin = uniform+comb = THE BENCHMARK** | **18.346** | **1.000** | **19.91** |
| retrim +52.5 (old seed) | 17.853 | 0.973 | 19.38 |
| best before retrim | 20.323 | 1.108 | 22.05 |
| band ±2% | 17.979-18.713 | 0.98-1.02 | 19.51-20.31 |
The old basin-1 seed at 0.9731 is OUTSIDE the new band ⇒ it HAD to be
reseeded (a campaign whose seed violates the guard trips/fail-closes).
NEW SEED, DERIVED from the two pitch-locked points (136296 +0 nm → 20.3226,
mx_retrim +52.5 → 17.8530; dW/dδ −0.04704 µm/nm, dT/dδ −1.423e-4 /nm):
**RETRIM_DELTA_NM 52.5 → 42.0** lands the seed at the band CENTRE 18.346 µm
with predicted T 0.9609 — i.e. the symmetric band makes the seed BETTER on
both axes (it stops paying T for width we cannot use). λ 1566.398 (DERIVED).
RELAUNCH: 136248/136249/136297 CANCELLED (≤3 evals each, zero accepted
improvements, ~11 GPU-h — cheap vs days under a superseded contract). New
labels avoid mixing FOM conventions in a resumed eval log:
**136465 = lumopt2_v2proj_s2 | 136466 = lumopt2_v2_uniform_s2 |
136468 = lumopt2_v2_noshift_s2**.
★DE-STEP VERDICT (job 136302 t24, MEASURED): removing the 46 nm boundary
step (teeth 20-25 ramped to the frozen 325) gives T 0.9600 @ 17.862 µm vs
the stepped 0.9594 @ 17.853 → **+0.0006, BELOW the 0.002 floor**. The step
is NOT the mechanism; the bulk inner-tooth κ is. BONUS: mean corr fell 1.75%
with width moving only +0.05% ⇒ the OUTER free teeth (20-25) are inert for
both T and width — the innermost teeth own the penetration depth. A
smooth, step-free geometry is therefore available at zero cost and is the
better default (fab-friendlier, no untested discontinuity at production N).
★COMB-OFF VERDICT (job 136302 t25, MEASURED): the comb removed from the
retrimmed best gives T 0.95652 @ 17.913 µm → 0.95639 at the retrim width,
vs 0.95941 with the comb ⇒ the comb is now worth **+0.0030 T** (1.5× the
0.002 noise floor). On the UNIFORM origin it was worth +0.0107. Its value
has fallen ~3.5× — the deeper inner teeth / wider cavity / shifts absorbed
most of it. Width impact +0.3% (neutral, as always measured). ⇒ 114 posts
for a gain barely above noise: a FABRICATION decision for the user, and the
comb-only optimization idea drops in priority accordingly.
USER DIRECTION 2026-08-23: production confirm at N≈169 NOT wanted now —
focus stays on the surrogate ("shorter") devices.
