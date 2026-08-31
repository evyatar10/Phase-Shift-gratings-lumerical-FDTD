---
name: lumopt2-design
description: Run, debug, resume, and extend the lumopt2 adjoint inverse-design program (currently the corr-325 pi-shift grating + SiN comb campaign). Use when the user asks to run/continue/check the inverse design, validate its cost function or gradients, dispatch a campaign, diagnose a lumopt2 failure, or set up inverse design for a new device.
---

# lumopt2-design — the inverse-design program runbook

> ## ★★★READ THIS FIRST, BEFORE ANYTHING ELSE IN THIS FILE
> **`runners/lumopt2_design/HANDOFF.md`** (full path:
> `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\HANDOFF.md`)
> is the current, self-contained state of the program as of 2026-08-18. Its
> section 0 carries the user's non-negotiables and its section 6b names the next
> experiment to run.
> **Why it overrides parts of this file:** `profile_line` was found never to have
> integrated over y, so **every σ and FWHM logged before 2026-08-18 is VOID**
> (T/λ/Q/R/loss are unaffected). Any width number quoted in this skill below,
> including in items 24-27, was measured through that broken path unless the
> handoff repeats it. The corrected width metric is the `post_processing`
> convention ONLY — the raw-line metric, the fitted FWHM_A_* slopes, and all
> coupled-mode-theory modelling were DELETED by user order; do not reintroduce.

LIVING DOCUMENT (user directive 2026-08-14): update this skill whenever the
program learns something — new gates, new lumopt2 bugs, campaign results,
new device families. It exists so a future session (or a new device) can
rebuild the whole workflow without re-deriving it.

## Where everything lives

- Engine + studies: `runners/lumopt2_design/` — `lumopt2_design.py` (engine),
  `validate_c325.py` (gates B0–B4), `campaign_c325_seedA.py` (Athena),
  `campaign_c325_seedB.py` (IGUM). One study = one runner; CONSTANTS at top.
- Box check: `runners/sweeps/tm_comb_box_c325.py` (gate A0 pattern).
- Deploy: `--lumopt2-design=<module>` flag in `athena/deploy_athena.sh` +
  `igum/deploy_igum.sh` (maintained pair) → `build_sweep_list.py` (one line
  per task, module `N_TASKS`) → `athena_run_one.py::_run_kind_lumopt2_design`
  → `module.main(task_idx)`.
- Decisions & measured state: `memory/project_inverse_design_cost_function.md`
  (physics contract), `memory/project_lumopt2_campaign_state.md` (live state),
  `memory/project_slurm_container_fixes.md` (cluster recipes),
  `memory/project_lumopt2_igum.md` (lumopt2 source analysis).

## The physics contract (settled — do not relitigate without the user)

- FOM = windowed high-p soft-max of Port_2 T (p=12, window ±2.5×measured FWHM
  re-selected every eval, stop-gradient on the selection). Q appears NOWHERE.
- Width anchor = analytic κ-ratio penalty ρ=Σcorr_d/(N_free·corr₀), asymmetric
  deadband +2 %/−5 % (β 18/5), injected by wrapping project.compute_fom/
  compute_gradient. Measured σ (2nd moment of the field_profile x-envelope) is
  a per-iteration TRIPWIRE only — NEVER in the adjoint. Hard guard 2κL ≥ 3.5.
- Params (~190): 25 free periods/side, (corr, avg, shift)/tooth, grating
  x-mirrored in the func; comb sites free per-site (r, x) + shared d,
  NOT x-mirrored (traveling 270° lattice). Optimizer L-BFGS-B; no global
  stage — 2 physics-informed seeds instead (uniform+comb winner; dip+overshoot).
- Surrogate N by 2κL ≥ 3.5 (corr-325 → N=100); width compared as ratio to the
  same-N control; winners get a §2 production confirm at N≈165-169 + accurate
  mesh OUTSIDE lumopt2 (plain SweepSpec runner).

## Validation pipeline (run in order; each gate is a hard stop)

| Gate | What / where | PASS |
|---|---|---|
| A0 | decorated-box check, 2 SweepSpec rows | judge on Q_i (≲0.5 % bias), NOT T |
| B0 | reader on stored .mat (local, 0 GPU) | measured ordering + linewidth-blind + penalty signs |
| B1 | build smoke + func-vs-builder diff (local) | <0.1 nm; shift contiguity exact |
| B2 | canaries through the full stack (cluster) | reproduce the LUMOPT2 in-study anchors; internal comb−bare ΔT matches family |
| B3 | validate_gradient, 6 params at a DETUNED point | 6/6 sign, α∈[0.8,1.25], vec-err ≤0.15 |
| B4 | known-answer mini-opt (comb δx 300→401) | >50 % recovery, ±50 nm |

Dispatch (Athena): `SBATCH_MEM=160G bash athena/deploy_athena.sh
--lumopt2-design=runners.lumopt2_design.validate_c325 --array-tasks=<n>`
(0=B2a, 1=B2b, 2=B3, 3=B4 — dispatch sequentially, gates between).

## Measured anchors (corr-325 campaign numerics: y6.8/z6.8, opt-region mesh)

- lumopt2 numerics are a NAMED §2 change vs the stored family (opt-region
  uniform-mesh override, unavoidable): λ −372 pm, Q −5.6 %, T unchanged.
  → in-study anchors (job 132631): bare T 0.9126/Q 1661/σ 18.378 µm/FOM 0.6757;
  seed-comb T 0.9233/Q 1670/σ 18.360/FOM 0.6839. Internal physics reproduces
  the family: comb ΔT +0.0107 (family +0.0105), Q_i +14.7 % (family +14 %).
- σ (2nd moment) ≠ fwhm_m (threshold FWHM) — never compare across observables.
- Pace: fwd+adj pair ≈ 26 min on H200; canary task ≈ 24 min end-to-end.

## lumopt2 (R1.3, 0.0.1.dev246) bugs + our fixes — check FIRST when debugging

1. `Project(project_name=…)` is DEAD — files go to a RELATIVE
   `lumopt2_project_<ts>` under CWD (in the container = ephemeral overlay,
   everything vanishes). Fix: set `project.fom.config_map.project_folder` to an
   absolute bind-mounted path BEFORE generate(). (Engine does this.)
2. `Box(...)` without explicit `dx/dy/dz` crashes addmesh ("Unsupported data
   type"). Always pass the mesh (50 nm = optimization mode).
3. SlurmRunner imports nonexistent `lumopt2.utils.lumslurm` → 2-line
   sys.modules shim (engine's `import_lumopt2` applies it).
4. Port expansion results have NO "T" key — read `|S|²` from "S".
5. NO checkpoint/resume; SIGTERM untrapped → our params/evals .jsonl logs +
   run_campaign cold-start resume are the recovery path.
6. ★THE BIG ONE — SOLVED 2026-08-16: lumopt2 dev246 resonant-FOM gradients
   are wrong ×5-29 because the true gradient is a TINY REAL PROJECTION of a
   huge complex sum Z (|Z| = 30-240× the gradient, arg Z ≈ 57-99°) and the
   pipeline's projection phase is off by ~6.7° (≈ the quarter-cell Yee source
   offset 0.25·k·dx = 6.2°). Per-class α = f(arg Z_class) — comb 1.3, corr
   5-8, shift 16, cavity 29, comb-d sign-flip; α is OPERATING-POINT dependent
   (comb-x flips sign between detune points) ⇒ per-class calibration is
   scientifically dead. Physics: λ-shifting params have ANTISYMMETRIC true
   dT(λ) (resonance translation, near-cancelling under the J window); the
   unfixed adjoint returns the QUADRATURE (symmetric lobe). FIX (engine
   `adj_phase_fix` + `adj_fix_re/im`): multiply the scaled adjoint fields by
   ONE measured complex C — **C = 0.8685+0.1022i fits FD on all 7 param
   classes to 1.7%** (residual per-param ≤10%).
   ★CALIBRATION RECIPE (new device / mesh / λ-window — C is not universal):
   (1) one naive validate_gradient (Re{Z} + FD, 16 sims); (2) one
   adjoint-only run with adj_fix=(0,1) (Im{Z}, 2 sims); (3) grid-fit
   FD_p = s·(cosφ·ReZ_p − sinφ·ImZ_p) over all params → C = s·e^{iφ};
   accept if residual ≤ a few % across EVERY class; re-verify at a second
   operating point. Detune points for any of this MUST sit inside the
   κ-penalty deadband, or subtract the analytic penalty from adj AND fd
   before fitting (detune-2 corr entries are contaminated by −3.204e-4).
   Historical: bc_patch ≤0.04% (TM walls are E∥-dominant), colocate never
   engaged as deployed (monitor recreated per eval — a setnamed after
   generate() is wiped; patch add_field_monitors instead if ever needed);
   both irrelevant post-fix. The earlier "α≈1.000" preview was a
   self-comparison artifact (retracted); `validate_gradient` returns
   **(fd, adjoint, err%)** — FD FIRST.
   MEASURED at both mesh refinements (jobs 132637/132657 — refinement ruled
   out; PVA kept anyway as recommended) + local layout-mode dEps probe proving
   the CAD side exact (|dEps| integral / analytic = 1.01-1.10; probe pattern:
   compute_opt_params_direct_to_permittivity_jacobian, dp is in PARAM units).
   NEVER dispatch a campaign whose gradients haven't been FD-validated (via
   the C-recipe above) at the current device/mesh/window.
7. `fom_symmetry_factors=[1]` is CORRECT for monitors centered on symmetry
   planes; ×2 factors apply only to monitors entirely on one side.
8. validate_gradient: run at a DETUNED interior point — the seed sits on the
   shift 0-bound (FD steps out) and AT the comb optimum (gradients ~0 there).
9. Login-node container CAD (fdtd-solutions) segfaults — read solved .fsp on
   compute jobs only.
10. n_params == 1 crashes the dp validator (length-1 squeeze → 0-d →
    shape[0] IndexError; jobs 132730/132735). Our `import_lumopt2` monkey-
    patches `DEpsCalculator._validate_and_normalize_dp` (atleast_1d fix).
    Also: a failing run() then crashes AGAIN in on_optimization_end
    ("Final FOM: None" format error) — always dig for the FIRST traceback.

## Gradient-fix experiment matrix (2026-08-15, user-ordered: try ALL routes)

Three fixes implemented in the engine, ALL spec-driven and FD-gated
(validate_c325 tasks 4-7; baseline = naive point-1 α from job 132657):
- Option 1 `spec.grad_cal` — per-class factors on the lumopt2 gradient
  (penalty gradient stays exact). Basis = measured α; viable only if task 4
  (second operating point) shows α stable ±30 %. Keep an FD tripwire.
- Option 2 `spec.bc_patch` + `spec.bc_eps_eval` — Johnson E∥/D⊥ correction as
  a NORMAL-component reweight of the per-component sparse dEps
  (BoundaryCorrected Parametrization subclass; walls are axis-aligned: widths
  → eps_y × R, shifts → eps_x × R; R = -Δ(1/ε)/Δε·ε_eval² = 0.537 clad / 1.10
  mid / 1.86 core — ε_eval is empirically arbitrated by the gate).
- Option 3 `spec.colocate_fields` — "nearest mesh cell" on optimization_dft
  (research: necessary, provably insufficient alone → expect tooth α 0.3-0.5).
Research digest + citations: memory/reference_adjoint_boundary_gradient_research.md.

## Docs/examples audit digest (2026-08-15 web sweep; full cites in the
## research transcript — public lumopt2 corpus = pyansys docs ONLY, zero
## third-party usage exists)

VALIDATED by the official examples: PVA mesh refinement (the L-bend example
sets exactly it), maximize-then-subtract sign, autograd func for use_jac,
explicit dp (auto-dp formula undocumented), finite bounds, fresh-Optimization
restarts, LocalRunner's built-in 2× layout-mode retry (= license-blip cover).
FIXED from the audit: custom callbacks list REPLACES the auto FileLogger (we
now add it explicitly); global monitor must be frequency-spaced (asserted —
PortResults snaps λ within 1e-9 m); optimization-region containment at BOUNDS
EXTREMES asserted in make_project ("the docs' loudest warning").
WATCH-ITEMS (not yet acted on): store_all_simulations writes .fsp AND
_output.h5 per iteration — estimate disk before a big campaign (300 GB quota
hang); our ports use "frequency dependent profile"=1 while the official
example sets 0 on both — divergence is benign per measured B2-B4 physics but
becomes the first suspect if adjoint anomalies appear; ftol is RELATIVE
(factr semantics) — keep FOM O(1); max_line_search arg exists only in the
R1.3-era module (R1.2 fallback would TypeError); validate_gradient's default
perturbation has undefined units — ALWAYS pass it explicitly; our fct's
detached index selection is unsupported-but-working — the FD gate is its only
safety net, keep it on every version change.

★UPSTREAM-FIX WARINESS (user, 2026-08-15): Ansys may fix/change these
internals in any future release. On EVERY Lumerical version bump: (1) re-run
the B3 gate UNPATCHED first — if tooth α ≈ 1 upstream fixed it, retire our
patches; (2) re-verify every monkey-patch still lands (dp validator, project
folder, SlurmRunner shim, bc subclass internals: compute_gradient_from_fields
/ calculate_dF_dPi signatures). Our patches assume dev246 internals.

## Turnaround efficiency (user priority 2026-08-15 — measured, then optimized)

MEASURED time budget per dispatch cycle this campaign: deploy/upload 1-3 min
(rsync is incremental; code deltas are KB) | queue wait 0 min-5 h (THE
dominant variable — h200-shared picked jobs up in minutes at some hours,
queued 5 h at others) | run = physics (necessary). So optimize CYCLES, not
uploads:
1. **One array per decision point** — every set of independent tasks rides a
   single deploy+queue cycle (the tasks-4-7 matrix pattern). Never serialize
   what one array can carry.
2. **Bracket uncertain knobs in the same array** — if a first guess (e.g.
   bc_eps_eval mid) fails its gate, the NEXT dispatch carries the whole
   bracket (core + clad) as siblings, not one-at-a-time cycles.
3. **Two-QOS lanes in the campaign era** — the long driver runs under 4d_1g
   while short tasks use 24h_1g; per-QOS running caps are separate, so both
   lanes progress concurrently. License seats (50) have never been binding —
   the QOS running-cap is.
4. **Two clusters** = the coarsest parallel lane (seed B on IGUM), seat-probe
   first (shared pool).
5. PROPOSED to the user, not adopted (CLAUDE.md §6 serialize rule is absolute
   as written): dispatching a NEW study while ONLY RUNNING (no pending) tasks
   occupy the queue is safe by the recorded mechanism — array tasks
   bounds-check sweep_list at task START, so running tasks are immune to the
   rewrite. Would unlock validation rows during multi-day campaign drivers.
6. Deploy-side micro-wins if ever wanted (small mirrored edits, not done):
   rsync -z over the VPN, consolidating the 5 rsync calls, a SKIP_SYNC knob
   for --array-tasks resubmits of unchanged code. Each saves ~1 min/cycle.

## The one-command program (user order: this is how you RUN it)

`bash runners/lumopt2_design/dispatch_campaign.sh seedA|seedB` — physics
params live as constants in the campaign runner files; the script carries
only cluster knobs (QOS/time/mem) + serialize-rule checks. General-memory
pointer: memory/reference_inverse_design_program.md.

## Campaign operations

- Driver = LocalRunner("GPU") inside ONE SLURM GPU allocation. GPU comes free
  via athena_run_one's lumapi monkey-patch (every FDTD session gets
  setresource GPU). SlurmRunner-driver mode is AVAILABLE on both clusters if
  per-sim jobs are ever wanted (see memory/project_slurm_container_fixes.md).
- **Walltime/QOS**: default 24h_1g kills >23:30 drivers; use 4d_1g-class QOS.
  ARRAY_TIME env override is silently IGNORED (conf overwrites) — use conf
  knobs and verify with `sacct --format=TimeLimit`.
- **Preemption**: every Athena partition REQUEUEs. run_campaign resumes from
  `{label}_evals.jsonl` on cold start (≈1 iteration lost). IGUM group
  partitions are PreemptMode=OFF.
- **★NO RE-DERIVING ACROSS LABELS (user rule 2026-08-30, "wasting me hours
  each time")**: a campaign that continues a toy/prior lane (same spec knobs
  + seed) must INHERIT its state — before dispatch, server-side copy the
  toy's `{label}_evals.jsonl` + `{label}_optstate.json` into the new label's
  out_dir so `_best_from_log` warm-starts from the toy's last accepted point
  (and the adaptive cap carries over); the dispatch note names the inherited
  rows. Never dispatch a separate seed/benchmark re-measure — if the seed's
  t_pk/λ/W exist in any stored eval log at the same numerics, cite them. The
  only legitimate seed forward is inside an optimizer iterate (its FIELDS
  feed the adjoint assembly; fields are not stored) — report it as
  "iterate-0 forward, fields needed", never as a benchmark run.
  ★Both edges (user, same day): a result's IDENTITY = engine version + §2
  numerics + spec params (cluster is NOT part of it — exact cross-cluster
  repro proven). A REAL identity difference (e.g. the R1.2→R1.3 engine bump)
  DOES warrant a re-run — name the differing component. But "I can't verify
  it's identical" is NEVER a reason to re-run: the stored jsonl / runner
  docstring / job log / HANDOFF carry version+numerics — read them first;
  re-run only on a FOUND difference, or unrecoverable provenance on a
  decision-critical number (say so explicitly). Label every stored result
  with its engine version + numerics so this check stays a 2-minute read.
- **License**: seats shared across clusters. Probe from IGUM before every
  multi-server phase: `$LUM/licensingclient/linx64/lmutil lmstat -a -c
  1055@132.68.48.51 | grep lum_fdtd_solve`; each campaign ≈ 2 concurrent
  seats; canary-first after any anomaly.
- **Scratch**: lumopt2 never cleans solver scratch (~15-20 GB steady per
  campaign label); stale validation `_files` dirs are deletable (ASK first).
- Serialize deploys per cluster (shared sweep_list); RAM 160G is ample
  (measured 6.5 GB for canaries).

## Scope (user, 2026-08-14)

Almost the entirety of this program is and will be the PI-SHIFT GRATING —
everything above is its contract. Other optimization targets may come later
(e.g. re-optimizing the grating coupler from the sibling repo
`grating_coupler_FDTD_codes`, or other devices) and those are NOT high-Q
resonant devices — their FOM/constraint physics is deliberately NOT specified
here (user: do not fill in unknowns in advance). What transfers vs what
doesn't is split below.

## Extending to a new target — what transfers, what must be re-derived

TRANSFERS as-is (device-independent):
- The lumopt2 wiring skeleton: builder-generated .fsp setup, Parametrization
  func over live object properties, custom autograd fct, project_folder /
  Box-mesh / shim / S-key fixes, callbacks + jsonl logging + cold-start resume.
- The validation METHOD: build-smoke func-vs-builder diff (B1 pattern),
  in-study anchors through the full stack (B2), validate_gradient at a
  detuned interior point (B3), a known-answer mini-opt on a measured axis (B4).
- All server ops: QOS/preemption/license/scratch rules (CLAUDE.md §6 +
  memory/project_slurm_container_fixes.md).

MUST BE RE-DERIVED per device (do NOT copy from the pi-shift contract):
- The FOM itself and its cheat channels — the soft-max-on-resonance reader,
  the κ-ratio width anchor, 2κL ≥ 3.5, surrogate-N, and Q_i auditing are
  HIGH-Q-RESONATOR physics; a non-resonant device (e.g. a grating coupler:
  broadband coupling efficiency) needs its own reader and its own
  anti-cheat constraint, settled with the user first.
- Seeds, bounds, parametrization basis, and the box-convergence criterion
  (Q_i-based judging is also resonator-specific).

## Campaign operations — measured facts from the first live day (2026-08-16)

11. **lumopt2 wraps fct exceptions TWICE** — scipy_optimizer.py:583 raises
    RuntimeError WITHOUT `from e` (original exception survives only in
    `__context__`), then optimization.py:852 re-wraps WITH `from e`. Any
    guard exception designed to cross opt.run() (RecenterNeeded, WidthTrip)
    must be recovered by walking BOTH `__cause__` and `__context__` — the
    engine's run_campaign does this now (both campaigns died once each to
    the naive catch: jobs 54309, 133016). Smoke any new guard end-to-end
    with a local replica of the double-raise before trusting it.
12. **Campaign disk = ~7 GB/iteration on Athena** (each fwd+adj solve leaves
    a 3.5 GB engine `*_output.h5` scratch dir next to its 25 MB .fsp; the
    KEEP_H5 cleanup of the array pipeline does NOT cover the lumopt2 path).
    A 60-iter campaign would eat ~450 GB → home-quota death mid-run (jobs
    silently hang at container init). Standing fix: `~/h5_roll_clean.sh` on
    the Athena login node (nohup loop, deletes campaign `*_output.h5` except
    the newest 2, every 30 min) — restart it after login-node reboots; check
    `quota -s` in every campaign health sweep. Only `*_output.h5` is ever
    deleted — .fsp/logs/jsonl are kept.
13. **Restart semantics (verified in code + live):** every cold start of
    run_campaign resumes from the HIGHEST-FOM row of `{label}_evals.jsonl`
    AND recenters the recording window on that row's λ (line ~771). Crash,
    preemption, walltime, guard-trip — all recover the same way, loss ≤1
    evaluation. To restart a campaign, just re-dispatch the same spec module;
    never rebuild anything by hand.
14. **The λ-drift direction is real physics in this family**: raising T at
    fixed width co-moves the resonance redward ~+1 nm per accepted early
    iteration, and LINE-SEARCH PROBES jump up to +2.6 nm (measured — three
    jobs died at the band edge in one day before the policy below).
    ★Gen-3 engine policy (2026-08-16): (a) a probe whose peak/FWHM leaves
    the recorded band gets a DEGRADED-but-finite FOM (full-band softmax —
    clipped peaks understate, so L-BFGS-B backtracks naturally; smoke: 0.204
    clipped vs 0.719 healthy, autograd flows); (b) RecenterNeeded fires ONLY
    when a BEST-so-far design drifts >2 nm from center (probes never trigger
    rebuilds); (c) MAX_RESTARTS=12. In-window evaluations are bit-identical
    to the gated physics — no §2 change. Window width itself stays
    §2-controlled; don't touch it without the user.

## Future-campaign candidates (user: "keep in mind" — none applied mid-flight)

- **Wider recording window**: ±5 nm @ 20 pm (501 pts) instead of ±3/301 —
  cuts recenter churn to ~1/campaign at slightly higher per-solve cost.
  Named §2 change (window+points ⇒ fresh anchors). DECISION CRITERION
  (2026-08-16): adopt for the next campaign IF this one's measured recenter
  frequency does NOT decelerate (still ~1 per 2 accepted iters by iter ~15).
- **Comb count/existence freedom**: density-comb stage (per-post index
  interpolation + binarization) or count ladder — see
  feedback_optimize_structural_counts + the count plan in campaign-state.
- **p-annealing** (broad-early/sharpen-late softmax) — only if a campaign
  stalls at high Q; reserved escalation from the high-Q methodology sweep.
- **Exact-C derivation**: chase the analytic origin of the adjoint phase
  constant (quarter-cell k·dx + amplitude) so new devices need no 2-sim
  calibration; also file/track the Ansys bug (evidence package banked).
- **H200 targeting at restarts**: measured 52 min/solve (A100, shared) vs
  9.6 min (H200) — at any planned warm-restart, check the H200 backlog first.

15. **★THE WIDTH-CHEAT (found live 2026-08-16, gen-4 closes it):** Σshift
    reconstructs the excluded cavity-LENGTH knob (cavity absorbs 2Σs by the
    walk's construction) → resonance detunes toward the stopband edge →
    mirror penetration ↑ → mode widens while ρ stays compliant (ρ models
    width only via κ∝corr — blind to detuning-driven penetration). Measured
    violator: all-25 shifts +5.1 nm mean ⇒ 2Σs +255 nm, λ +2.6 nm, σ +9.6%,
    T +0.02. GENERAL LESSON for any parametrization: enumerate the LINEAR
    COMBINATIONS of allowed knobs that reconstruct excluded ones (here:
    sum-of-shifts = cavity length) and guard them analytically — a measured
    tripwire alone recovers but doesn't teach the optimizer; put a
    differentiable wall (elongation penalty, deadband 120 nm) so L-BFGS-B
    feels it. ALSO: any best-row restart selection MUST filter on constraint
    compliance — cheat designs are FOM-best by construction, and an
    unfiltered argmax restarts inside the violation (measured burn loop).
    Keep violator rows in the log: they measure the constrained trade
    (+0.02 T per +10% width at the band edge — writeup material).

16. **★LOADED-vs-DISK CODE DIVERGENCE (found live 2026-08-17):** a constant
    tightened on disk mid-campaign (RHO_UP 1.02→1.01) does NOT reach a
    running driver — Python never reloads modules, and in-process guard
    restarts reuse the loaded module too; only a JOB-level restart picks
    up new code. Measured consequence: seedB accepted σ-ratio 1.0121 with
    no trip (loaded band 1.02). The DANGEROUS part is retroactivity: the
    restart-selection filter applies the NEW constant to the OLD log, so
    every best row accepted between the push and the eventual reload gets
    silently discarded on restart — rollback loss GROWS with time. RULE:
    any guard/threshold change during a live campaign is incomplete until
    either (a) the affected jobs are deliberately restarted (user-approved
    scancel), or (b) the not-in-effect status + growing rollback exposure
    is reported to the user the same session with a restart recommendation.
    Never state the new value is "active" while any launched-before job
    still runs. (Also re-chain any afterok dependent when restarting — a
    dependent of a cancelled job pends forever.) RESOLUTION (user,
    2026-08-17): option (c) chosen — revert the DISK value to the loaded
    one (RHO_UP back to 1.02) for program-wide consistency; zero progress
    lost, hazard eliminated. Width honesty moved to the readout layer:
    Q_i/σ² (the width-immune metric; it kept rising 216→224→235 through
    the first walled steps = gains genuine) + fixed-width production
    re-trim. The 1.01 tightening is SUPERSEDED — do not re-tighten
    mid-campaign; revisit only between campaigns if a delivered design
    pins the +2% wall.

20. **★FD-STEP-vs-SLIVER-BOUNDS trap (killed stage-2 job 133499 at 1h51;
    latent in the bare campaign, never exercised):** lumopt2's dEps
    calculator central-differences EVERY parameter with the spec dp
    (1.0 nm) and RAISES when 2*dp exceeds the param's bound range —
    frozen blocks with ±1e-3 slivers cannot fit it. B2-style canaries
    never catch this (compute_fom only, no gradient). STRUCTURAL FIX in
    make_project: per-param clamp dp_i = range/4 when 2*dp_i >= range —
    frozen params get ~5e-4 nm steps whose dEps is below mesher
    resolution (gradient 0 = the meaning of frozen); active params
    untouched. Smoke: assert no param has 2*dp >= bound range for every
    new spec family. GENERAL: any new frozen-block mechanism must be
    exercised through ONE GRADIENT computation before a campaign trusts
    it — a forward-only canary proves nothing about the dEps path.

19. **★THE ELONGATION WALL IS CORRECT AT 120 nm — do NOT relax it when the
    campaign plateaus there (settled on Fable, 2026-08-17 morning).** Both
    independent seeds walked 2Σs to ≈130-140 nm and stalled width-COMPLIANT
    (+1.5/+1.7% vs the +2% band), which LOOKS like the proxy binding ~15-20%
    tighter than the spec. But the gain available past the wall is width-
    bought by construction: riding to the true +2% limit buys only ≈+0.002 T
    (interpolated seedB best→probe11) with Q_i/σ² FLAT — fake gain for the
    fixed-width claim — and the measured 2-3× shape-sensitivity (item 18)
    makes a looser sum-wall less safe than nominal. A plateau at the wall =
    genuine convergence of the shift direction, NOT guard suppression. The
    correct response is STAGE-2: restart from the compliant best with the
    SHIFT BLOCK FROZEN at its discovered values (sliver bounds, same
    mechanism as frozen combs — use replay_params + a bounds override), so
    all solves go to corr/avg/comb/cavity where genuine gains live. Physics
    unchanged, no channel re-opened, existing restart machinery.
    Seed-value fact for the writeup: uniform start T 0.8924→0.9328 unaided;
    dip seed 0.9381→0.9460 ⇒ the physics-informed seed ≈ +0.046 head start.

18. **★Σshift is an IMPERFECT width proxy — SHAPE matters, not just sum
    (candidate, observed live 2026-08-17 seedB evals 10→11).** A move that
    raised 2Σs by only +22.9 nm widened σ by +0.308 µm (0.0134 µm/nm),
    while earlier compliant steps gave 0.004-0.008 µm/nm — a 2-3x higher
    sensitivity for the same elongation. Corrugation was NOT involved (mean
    321.78→321.67 nm, ρ compliant), so this is not a κ-redistribution
    loophole; what changed was the shift PROFILE SHAPE (inner teeth pulled
    back 1.90→0.26 nm while outer grew), i.e. a chirp of the local Bragg
    phase that alters penetration depth independently of the total. NOT
    proven (few points, possibly nonlinear relation) — a controlled scan at
    fixed 2Σs with varying shape would settle it. CONSEQUENCE: the
    analytic elongation wall cannot be the only width defence; the
    MEASURED-σ layer is what closes shape-driven channels, and this is
    concrete motivation for the v2 σ-gradient FOM. No damage occurred —
    the probe was FOM-rejected (0.7000 vs best 0.7004) before the tripwire
    was needed, which is the layered design working.

17. **★SLIVER-BOUNDS TRAP when evaluating an EVOLVED vector under bare /
    frozen-comb specs (burned 2026-08-17, job 133395 task 1, 34 s):**
    `param_bounds` pins the comb slots to `(seed ± 1e-3)` whenever
    `bare=True` or `free_comb=False` (they are inert — func emits no
    scatterer properties). Feeding a campaign's EVOLVED params there dies
    with `ValueError: Parameter 75 value ... outside bounds` before any
    solve. FIX for any A/B or replay of evolved params under a bare spec:
    reset the comb block to `seed_params(spec)` values first (physics-neutral
    — the comb is absent from the scene) and keep the grating block
    untouched. Add a bounds-compliance smoke (`all(lo <= p <= hi)`) plus a
    grating-identity assert to any runner that replays stored params.
    GENERAL: a "frozen" parameter block is frozen AT THE SEED, not at
    whatever the caller passes. ★STRUCTURAL FIX (2026-08-17): the engine now
    exports `replay_params(spec, p)` — resets inert comb slots to seed under
    bare/frozen specs and asserts full bounds compliance. EVERY runner that
    replays stored/evolved params MUST go through it (comb_dip_ab.py is the
    reference usage); never hand-roll the reset again.

21. **★BOUNDS WIDTH IS A LEARNING RATE — read from lumopt2 source 2026-08-17**
    (`optimizer/scipy_optimizer.py`): every parameter is scaled to [−1,1] by
    its OWN bounds via `ParameterScaler(target_range='centered')`, and the
    gradient is transformed `g_scaled = g_physical × range/2`. Consequences,
    all of them load-bearing for an automatic platform:
    (a) **a parameter's effective step size is proportional to its bounds
    width** — widening a bound to "give the optimizer room" silently
    multiplies that block's influence on the search direction. Bounds are a
    NUMERICAL choice here, not just a physical one; set them per block with
    that in mind, and never compare raw physical gradient components across
    blocks (compare `g × range/2`).
    (b) sliver-freezing (item 17/20) works *because* it drives that block's
    scaled gradient to ~0 — the freeze is a scaling effect, not a hard
    constraint, so a frozen block can still drift within its sliver.
    (c) the MEASURED comb flatness is therefore real, not an artifact: comb
    scaled gradient ~1e-4 vs shifts/cavity ~7e-2, i.e. 500× smaller AFTER
    the range weighting (comb-x range 200 nm actually *amplifies* it).
    (d) **the x0 duplicate-eval tax (stage-3 133541, ~1.7 GPU-h) —
    ★CONFIRMED 2026-08-17 ~20:55: eval 3 took a real (in fact huge) step, so
    eval 2 was the duplicate, NOT the v1 zero-step failure:** lumopt2
    logs its own `Iteration 0 (baseline)` and then hands x0 to scipy, which
    evaluates f(x0) again. The scaler round-trip (physical→scaled→physical)
    returns the vector 5e-15 off unless the value sits exactly at the
    bound MIDPOINT — so frozen/sliver blocks round-trip exactly and the
    duplicate is free, while free blocks miss the exact-match cache and pay
    a full forward+adjoint on a physically identical device. Budget one
    extra evaluation per campaign start, or centre bounds on p0 to dodge it.

23. **★THE COMPLETION PATH HAD NEVER RUN — `opt.run()` returns a TUPLE
    (measured 2026-08-17, IGUM bare 55343).** `run_campaign` read
    `result.final_fom`; lumopt2 R1.3 returns `(params, fom)`, so the FIRST
    campaign in the program's history to reach natural completion died with
    `AttributeError` after finishing all its physics. Every earlier campaign
    was stopped, cancelled or crashed mid-run, so the last ~10 lines of the
    main entry point had literally never executed. FIXED: `_final_fom(result)`
    accepts object/tuple/list, degrades to -inf on an unknown shape (the value
    is bookkeeping only — the delivered design always comes from the
    width-filtered log). GENERAL LESSON, worth more than the bug: **the code
    that runs ONCE AT THE END of a long job is the least-tested code you own.**
    Exercise finish/teardown/summary paths with a 2-minute toy run before
    trusting them at the end of a 10-hour campaign. Loss here was cosmetic
    (only `<label>_best.json`) ONLY because the per-eval jsonl is written by
    the callback — keep it that way: never make the summary file the only
    place a result lives.

22. **★BOUNDS ARE THE TRUST REGION — set them per RESTART, not per physics
    (measured twice on 2026-08-17: stage-3 133541 eval 3 and bare 55343
    eval 3).** L-BFGS-B's first step is UNIT-NORM IN SCALED SPACE, and item
    21 says scaled space is bounds-normalized — so on a warm start every
    wide-bounds block gets slammed by a fraction of its FULL RANGE on the
    very first probe, no matter how good the seed is. MEASURED: shift bounds
    (0,200) → first probe moved 2Σs 130.6 → **504.2 nm** (3.9×, up to 9.9 nm
    on a single tooth), σ 17.749 → **19.888 µm** (+13.7%, band is +2%),
    FOM 0.6897 → **−7.92**. The bare campaign did the same thing on its own
    first free step (σ 21.3 µm).
    ★★SEVERITY UPGRADE (measured hours later, same day): this does NOT merely
    waste ~1.7 GPU-h per probe — **it can KILL the campaign.** IGUM bare 55343
    ended with `ABNORMAL_TERMINATION_IN_LNSRCH` after exactly ONE accepted
    iteration: the blow-out threw the line search so far off that maxls=4 was
    exhausted before the Wolfe conditions could be met, and L-BFGS-B gave up.
    Four hours of solves produced nothing after 18:45. The day's whole pattern
    reduces to this ONE mechanism: stage-3 overshot (cancelled), bare overshot
    (died), stage-2 climbed cleanly for 8 h — because its frozen-shift slivers
    were, by accident, exactly the trust region the other two lacked.
    => `trust_nm` is not hardening, it is what makes a free-shift campaign
    VIABLE. Any campaign that unfreezes a block MUST carry it.
    **RULE for any warm-started campaign: set each free block's bounds to
    p0 ± (the step scale you actually want), not to the physical limit.**
    Stage-2 got this right BY ACCIDENT (frozen shifts = a 1e-3 nm trust
    region) and is the run that made clean monotone progress. Corollary:
    the physical limit still belongs somewhere — enforce it in the penalty,
    which is differentiable and re-anchors, not in the box.
    ★ENGINE FIX SHIPPED (2026-08-17, Fable decision): `CampaignSpec.trust_nm
    = {"shift": 20, ...}` clamps named blocks to p0 ± r CENTERED (r shrinks
    near a physical edge; seeds ON an edge keep the plain box). Centering
    makes the bounds-scaler round-trip bit-identical → ALSO kills the 21d
    duplicate-x0 tax. Opt-in, default None → inert for every existing spec
    (smoked: stage-2/3/AB bounds byte-identical; REQUEUE-resume safe).
    OPERATIONAL DECISION same session: stage-3 (133541) was CANCELLED
    rather than restarted — its stage-1 seed (T 0.9318) had been overtaken
    by stage-2 (0.9609), so the tangent walk from there could no longer
    reach the frontier, and its 160G blocked the comb scan's second slot.
    The tangent question re-launches FROM THE STAGE-2 WINNER when stage-2
    plateaus/trips, with trust_nm ON and sig_anchor re-measured on the
    winner row. ★v2 (banked, principled fix): σ̂ is LINEAR in p, so the
    right tool is a linear inequality constraint + a constrained method
    (SLSQP/trust-constr project the search direction ONTO the σ-neutral
    tangent — exactly the wanted physics, no wall collisions at all);
    lumopt2's ScipyOptimizer does not expose scipy's `constraints` arg, so
    it needs an optimizer subclass in the engine — v2 work, not mid-flight.
    Second measured caveat from the same event: **the linear σ̂ surrogate
    UNDER-predicts at large excursions** — at 2Σs +374 nm (4.7× outside its
    fit range) it predicted 19.264 µm vs 19.888 µm measured, i.e. it errs
    toward under-penalizing. Fine while the penalty is huge anyway, but do
    not trust σ̂ as a guard far outside its fitted neighbourhood.

- **★USER DIRECTIVE (2026-08-16): develop a v2 cost function with a real
  σ (mode-width) gradient.** ★2026-08-17 addendum — it will NOT move the
  comb: the comb is flat in σ too (removing it entirely moves σ by 0.04%,
  17.7045→17.7120 µm, MEASURED), so a σ-adjoint hands it a second ~zero
  component. The comb-side lever is a REPARAMETRIZATION — replace 57
  independent site-x with 2 collective coordinates (global phase, pitch);
  the collective derivative is the SUM of 57 individually-at-noise terms,
  which can be measurably non-zero. Both remain LOCAL: the basin question
  needs the scan (job 133718). Routes assessed: (a) validate lumopt2's
  FieldResults adjoint for a second-moment functional (the C-recipe applied
  to the field-adjoint path; ~a day + FD gate) — the in-toolchain path;
  (b) eigen-solver (FEM/QNM) stage where width derivatives come from
  eigen-perturbation — different toolchain, use as winner cross-check;
  (c) REJECTED: LDOS substitution (the literature's differentiable Q/V
  trick) — LDOS ∝ Q/V conflates Q and V, so a fixed-width-while-Q-improves
  constraint would punish legitimate radiation reduction; only valid for
  joint Q/V maximization. (d) REJECTED: CMT width model — tooth-scale
  optimizer moves violate slowly-varying assumptions (user physics call).
  ★Scope note (user, 2026-08-31): the CMT ban applies INSIDE the optimizer /
  width-wall only. The standalone q3db PREDICTION program (python_tools/
  bragg_cmt.py + calibrate_q3db.py + predict_q3db.py, memory
  project_q3db_predictive_engine.md) is user-authorized to use CMT and is
  backtested; do not import its width laws back into lumopt2 surrogates.


## ═══ PLATFORM RECIPE — distilled 2026-08-18 (Fable handoff) ═══

The end-state goal (user): ONE program that runs the whole optimization with
no human decision points. What two days of live campaigning proved is NEEDED
vs NOT NEEDED:

**The automatic pipeline (in order):**
1. Anchors: one canary forward per family (B2-style) -> sigma0, lambda0,
   T0 vs stored controls. Never re-run stored controls (cite them).
2. Decorations are PRE-COMPUTED, not co-optimized: comb pitch from the
   grating equation lam/(n_eff + n_clad*|u_x_needle|) (531 for this family;
   light-line cutoff = the design's hard floor — stay >= ~2 nm above it),
   phase 270 deg, r 80 (flat 70-100), d 1.9 um, length ~ mode-length-matched
   (k-space: comb beam width 1/L_comb ~ needle width 1/L_mode). Verified by
   a one-time basin scan (9 forwards); the adjoint then confirms it stays
   motionless — do NOT spend campaign DOF on it.
3. ONE campaign, everything free, from the known-good seed, with:
   - trust_nm on every free block (bounds ARE the first-step size; centered
     on the start point; the engine re-centers per attempt/resume — items
     21/22). No freeze stages needed anymore: stage-2's freeze was only an
     accidental trust region.
   - sigma-hat wall (single hinge on the calibrated linear width surrogate,
     re-anchored each restart) + the measured-sigma cumulative tripwire band
     as the outer guarantee. NEVER twin walls (they forbid the sigma-neutral
     cross-block trades where the real gains live).
   - completion-path toy run before the long dispatch (item 23).
4. When the marginal step efficiency (dFOM per um of width spent) collapses
   ~100x below the shift lever's 0.065/um, the stage is DONE in that
   subspace — re-seed a fresh stage from the best width-compliant row
   (stage-wise restarts beat one long run: re-anchoring + re-centered trust
   regions + fresh L-BFGS memory each time).
5. Close-out (the only reportable numbers): scale-check ladder on the winner
   (shift x0/x0.5/x1.5 - catches stage-1 legacies), decoration-removed row,
   then production confirm at N~165-169 accurate mesh + lock-target re-trim.

**Measured NOT-needed (do not rebuild these):** PSO/global stage; comb in
the adjoint loop; wide-tooth-length/duty-cycle DOF (duty slaved to shifts,
kappa flat to 0.02% over the full shift range - sin(pi*D) max at D=0.5);
sigma-derivative for the comb (comb is flat in BOTH T and sigma); parallel
freeze-stage ladders.

**Transfer law for Q projections (validated to +1.3% on the control):**
Q_i_production = Q_i_surrogate x (mode_prod/sigma_surrogate)^2; at -3 dB,
Q_loaded = (1-sqrt(T))*Q_i = 0.2953*Q_i. Current best projects ~41,000
(EXPECTED, only the accurate-mesh confirm is reportable).

24. **★THE sigma-hat SURROGATE DOES NOT TRANSFER BETWEEN BASINS, AND ITS
    ANCHOR NEVER REFRESHES INSIDE A STAGE (measured 2026-08-18, seedB2 job
    56033).** The wall's coefficients (SIG_A_SHIFT 0.00368/nm, SIG_A_RHO
    -3.85, SIG_A_WCAV 0.01) were fitted on SEED A's device. On seed B's
    profile (dip 234 nm + 13 teeth of overshoot) they OVER-predict badly as
    soon as the design moves:
        ev1 err +0.005 | ev2 +0.008 | ev4 +0.014   (near anchor: fine)
        ev3 err +0.604 | ev5 +0.318                 (two steps out: broken)
    CONSEQUENCE MEASURED: ev5 was T 0.9591 at MEASURED ratio 1.0198 — i.e.
    genuinely IN BAND and better than seed B's best — but sigma-hat claimed
    18.158 um, penalty 0.528, FOM 0.1835 => **FALSE REJECTION of a compliant
    design**. The run is fenced into a small neighbourhood of its seed by a
    model that is wrong outside it.
    ROOT CAUSE OF THE NON-SELF-CORRECTION: the measured-sigma tripwire and
    recenter guard fire ONLY on ACCEPTED-BEST designs (deliberate — a probe
    must not restart the campaign), and the anchor is re-zeroed ONLY on
    restart. So a stage whose probes keep getting rejected never re-anchors,
    and the surrogate error compounds exactly where accuracy matters most.
    ★FIX (principled, not yet applied — needs a restart, so PARKED for the
    user): re-anchor the sigma-hat wall on EVERY ACCEPTED ITERATION using
    that iterate's measured sigma, not just at restarts. The surrogate is a
    LOCAL linear model; its anchor must track the current point. Both of
    tonight's false rejections would have been avoided.
    ★WIDER LESSON for the platform: any fitted surrogate standing in for a
    quantity you can measure per-iteration must be re-fitted or re-anchored
    at the measurement cadence — otherwise it silently becomes a constraint
    on the OPTIMIZER's imagination rather than on the DEVICE's physics. This
    is also the strongest argument yet for the v2 sigma-adjoint (item above):
    a true derivative has no basin-transfer problem.

25. **★★THE PROXY TRAP — WE CONTROLLED sigma FOR A WHOLE CAMPAIGN WHILE THE
    SPEC WAS FWHM (measured 2026-08-18, job 134217 vs 134107).**
        uniform ORIGIN : T 0.8926, sigma 17.487, FWHM **17.100**, ratio 0.978
        optimized best : T 0.9659, sigma 17.818, FWHM **22.210**, ratio 1.247
    **sigma +1.9% while FWHM +29.9%.** The +2% sigma band was satisfied at every
    single step and the spec observable still grew by a third. Cause: sigma is a
    SECOND MOMENT and is blind to a FLATTENING CORE — the optimizer widened the
    half-max width while arranging the tails so the moment barely moved. The
    FWHM/sigma ratio going 0.978 -> 1.247 IS that shape change.
    ★THE GENERAL RULE (the reason this is item 25 and not a footnote): **never
    let the CONTROLLED quantity differ from the SPECIFIED quantity without
    measuring both on every evaluation.** A proxy is only a proxy while the
    shape that links them is fixed — and an optimizer's whole job is to change
    shapes. If the spec says FWHM, either constrain FWHM or prove per-eval that
    the ratio holds. We did neither for two days, and every "in band" claim in
    DESIGNS.md before this date means IN THE SIGMA BAND, nothing more.
    ★SHIPPED SAME DAY (both alarms verified against the real numbers):
      - every eval logs `mode_fwhm_um`, `fwhm_over_sigma`, `sigma_hat_um`,
        `sigma_resid_um`;
      - `[MODE SHAPE DRIFT]` fires when FWHM/sigma moves >0.05 from the origin's
        0.978 (i.e. when sigma stops proxying the spec);
      - `[WIDTH-SURROGATE OFF]` fires when the wall's prediction misses the
        measurement by >0.02 um (item 24's failure, now self-announcing).
      - sub-lesson: the shape alarm was FIRST written behind the surrogate's
        early-return and logged None on its own audit row. A diagnostic must not
        depend on whether an unrelated feature is configured.
    ★OPEN when this was written: FWHM rows for BEST_T9635 and for the
    shifts-zeroed control (134217 t1/t2). The control decides the response — if
    zeroing the shifts restores FWHM ~17.1 the broadening is shift-driven and
    the same lever fixes it; if it stays ~22 the corr/cavity shaping did it and
    the constraint must be rebuilt on FWHM.
    ★UPDATE (134217 t1 landed): best re-measured T 0.9640 / sigma 17.800 /
    raw-FWHM 21.709 — the +27% growth is double-measured, the finding stands.

26. **★★FWHM HAS TWO CONVENTIONS IN THIS PROJECT — NEVER COMPARE ACROSS THEM
    (2026-08-18, user caught it: "original was ~19, not 17.1").**
    - RAW-LINE (engine's first `mode_fwhm_um`): absolute half-max from zero on
      the oscillating |E|^2 line. Origin reads **17.100**.
    - PROJECT convention (`post_processing.fwhm_m`, every stored study, the
      nladder's 19.24 um, the ~20 um spec): `extract_envelope_peaks` (cubic
      through standing-wave peaks) + `calculate_fwhm_relative` (half-max
      RELATIVE TO THE PROFILE FLOOR), on the y-INTEGRATED profile. The same
      family reads ~19+ here. Also: 19.24 um is the BARE N=100 device — the
      comb-decorated origin was never measured in this convention until now.
    Consequence: quoting 17.1 next to the 19.24/19.91 anchors was a
    convention-mixing error.
    ★★AND THE RAW-LINE METRIC IS NOT SAFE FOR RELATIVE CHANGE EITHER (realised
    2026-08-18 12:45, before acting on it): first/last crossing of an ABSOLUTE
    half-max on an OSCILLATING standing wave moves with the FRINGE CONTRAST and
    the node floor, not only with the envelope. Independent coupled-mode theory
    (int_0^{x_h} kappa dx = ln2/2; reproduces the stored origin 19.24 um to 2%)
    predicts only ~4% width growth where raw-line reported 27% and sigma 1.8%.
    Three estimators, three answers => the magnitude was UNKNOWN and the fix
    (measure the ENVELOPE convention) had to land before any campaign was
    cancelled on the strength of it. **Never restructure a program on a number
    from a metric you wrote the same day and have not cross-checked.**
    ★FIX (shipped, engine): `profile_line` fetched once per eval; THREE
    metrics logged (`sigma_um`, `mode_fwhm_um` raw-line for continuity with
    the 08-18 audit rows, `fwhm_env_um` = the project convention = the spec
    observable); the raw (x, |E|^2) line SAVED to `<out>/profiles/*.npz`
    (~30 kB/eval) so any future metric question is answerable OFFLINE — the
    audit needed GPU re-runs only because no profile was ever kept.
    ★GUARD: `CampaignSpec.fwhm0_um` — when set, accepted-best designs must
    hold `fwhm_env_um/fwhm0_um` in the same +2%/−5% band (WidthTrip), and
    `_best_from_log` filters restarts/final selection on it. Default None =
    legacy sigma-only (live campaigns unaffected by a REQUEUE).
    ★USER CONSTRAINT (2026-08-18, verbatim intent): re-matching the UNIFORM
    corrugation to drag FWHM back to ~20 **does not count** as fixing this —
    the origin's uniform corr stays as-is; the optimization must win T while
    genuinely holding the spec observable.

28. **★★V2 WIDTH GRADIENT — VALIDATED OFFLINE 2026-08-21, see
    `runners/lumopt2_design/V2_FWHM_PLAN.md` (the v2 spec; supersedes the
    "route (a)" sketch in the σ-gradient directive above).** STORM research
    (4 perspectives) + zero-GPU validation on the 7 corrected profiles + 3
    stored families produced: (a) **`softW`** — soft superlevel-set width on a
    boxcar(258nm)+Gaussian(0.25µm)-smoothed y-integrated line, floor-relative
    half level — **tracks measured `fwhm_env` to ≤2 pp** where σ errs 24 pp
    and the participation ratio 21 pp (both L²-moments PERMANENTLY excluded);
    autograd gradient ≡ FD to 1e-8; LOCAL surrogate (−8 pp at N-ladder-scale
    excursions) ⇒ re-anchor to measured fwhm_env EVERY accepted iterate.
    (b) lumopt2 R1.3 `FieldFom` (read from local source) supports ONLY per-λ
    Σ|E|² scalars with a hard-coded conj(E_fwd) adjoint source ⇒ a width
    adjoint needs a subclass importing the WEIGHTED source W(x,y)·conj(E_fwd),
    W = autograd dF/dI × y-trapz weight; +1 adjoint/iter; the field-adjoint
    path needs its OWN C_field calibration + FD gate (W3) — never assume the
    port C transfers. (c) Architecture: augmented Lagrangian
    over L-BFGS-B — chosen on GENERAL optimization grounds (fixed penalties
    leak or ill-condition; AL multipliers driven by MEASURED violations; no
    optimizer swap so items 13/21/22 stay valid; CCSAQ/trust-constr banked
    as fallback if multipliers oscillate 2 outer cycles). ★"SPINS" in early
    v2 notes was a mistranscription of STORM (the research METHOD, user
    2026-08-21) — SPINS is background corroboration only, nothing rests on
    it. Filter acceptance, measured re-trim at stage boundaries (HANDOFF §6
    projection),
    see-saw-seeded start. Gate ladder W0-W6 in the plan file; W0 passed
    2026-08-21. ★IMPLEMENTED same day: `CampaignSpec.width_grad` + softW +
    `make_width_classes` (MixedFom) + AL penalty + per-eval softw/fwhm_hat
    logging + per-restart re-anchor/multiplier updates; default off = every
    existing spec bit-identical. Local W0/W1 gates PASS (autograd≡FD 6.6e-7
    after fixing a detached-normalizer bug the gate itself caught — never
    detach a value the softmax weights depend on). W3 (cluster FD gate with
    C_field) + W1 toy completion run still MANDATORY before any campaign;
    W3 must also verify the port source is disabled in the width-adjoint
    .fsp and the single-λ source import zeroes other planes.
    ★DISPATCH LESSONS (job 135954, all 4 tasks dead in 35 s, zero GPU lost):
    (a) lumopt2 generate() REJECTS broadband FieldRegion monitors
    (`_verify_not_broadband`) — a width FOM needs a SINGLE-λ twin monitor
    (`field_profile_adj`, built by build_base_fsp when width_grad); and with
    "override global monitor settings" on, **`use source limits` must be
    set 0** or the validator reads the SOURCE span and still rejects.
    (b) `trust_nm` block keys are corr/avg/shift/r/x/d/**wcav** — "cav" is
    a KeyError. (c) ★NEW MANDATORY LOCAL GATE: run a silent LOCAL
    `project.generate()` (lumopt2's session is hidden by default) before
    ANY dispatch that changes the FOM/monitor/spec configuration — it
    reproduces the whole generate()-time validation class in ~3 min with
    zero cluster cost (it caught the use-source-limits bug immediately).
    Redispatch after fixes: job 135971. (d) SECOND run-path trap (135971
    task 12, 53 min): PortFom's `_get_port_monitor_info` AND the 'adjoint'
    branch of `_update_port_positions` loop over ALL config-map entries as
    `FDTD::ports::<name>` — MixedFom must filter width entries out of BOTH
    (shipped). Invisible to local generate(); the run-path port loops are
    only exercised on cluster — expect this class whenever PortFom grows a
    new all-entries loop in a future lumopt2 version.
    (e) ★★THE FIELD-FOM MONITOR MUST BE AN `addfieldregion` OBJECT (found
    2026-08-22, cost 135986's three forwards): in R1.3 build 4572 NO plain
    monitor type has 'source mode' (measured: profile/power/time all lack
    it) — stock FieldFom's setnamed(monitor,'source mode',True) works ONLY
    on the dedicated FieldRegion object (type "FieldRegion", addfieldregion;
    monitor+adjoint-source hybrid, own λ controls). A DFT-monitor copy
    passes generate() and the forward, then dies at adjoint setup. ALSO:
    source-disable for ('port', name) entries needs the full
    `FDTD::ports::<name>` path. Local gate W1.5 (in the generate smoke)
    now exercises the adjoint-setup property sequence on the built scene —
    run it before any field-FOM dispatch. Redispatch: job 136026.
    (f) ★GPU ENGINE REJECTS FieldRegion-SOURCE adjoint scenes ("ERROR:
    invalid configuration argument", CUDA kernel-launch level — measured
    136026 all 3 tasks; docs list only TFSF/BFAST as GPU-unsupported, the
    FieldRegion object is undocumented for GPU). FIX: WidthAwareRunner
    (engine, auto with width_grad) routes jobs whose path contains
    field_profile_adj to spec.wg_adj_resource="CPU"; all other solves stay
    GPU. CPU adjoint solve time = measured by W1r; if >~1.5 h/solve the
    campaign iteration cost needs a user decision (Ansys report / accept /
    rethink injection). Redispatch: 136035 (24h_4g lane for CPU headroom).
    ★MEASURED same session (seed_width_audit task 0): rebuild-from-logged-
    params reproduces the logged T to 1e-6 — param-vector replay is EXACT.

29. **★QUOTA KILLED A JOB BECAUSE THE ROLL-CLEANER'S GLOB WAS STALE
    (2026-08-23, job 136090 died "Disk quota exceeded" at 4:17).** The v1
    `~/h5_roll_clean.sh` globbed ONLY `results/campaign_c325_*`, so every
    study created since (lumopt2_v2proj, validate_c325, retrim_best,
    retrim_decompose, seed_width_audit...) accumulated 3.5 GB `*_output.h5`
    per solve UNWATCHED — home reached the 330 G HARD limit. Shipped v2:
    walks ALL of `results/`, keeps the newest 2 per directory, never touches
    files modified <30 min (active solves), 15-min loop. **GENERAL RULE: any
    janitor keyed to a NAME PATTERN rots the moment a new study is named
    differently — key janitors to the FILE TYPE and a recency guard, never to
    a study-name glob.** Second lesson: `validate_gradient` launches its FD
    legs CONCURRENTLY (14 sims => ~49 GB of scratch at once) — check quota
    headroom before dispatching a gradient gate, and prefer fewer indices
    when a campaign is already running.

30. **★A MULTI-ENTRY FOM COSTS ~25 GB MORE THAN A SINGLE-ENTRY ONE — SIZE
    MEMORY BY ENTRY COUNT, NOT BY SIMULATION COUNT (job 136122 OOM-killed,
    exit 137, 2026-08-23).** `base_fom.calculate_gradient_fields`
    (base_fom.py:473-511) holds the forward region field array for EVERY fom
    entry simultaneously (phase 1), then each entry's adjoint array (phase
    2); each is (nx,ny,nz,3)×n_wl complex128 ≈ 50 MB per λ for a 25-period
    optimization region. A MixedFom (port T + width) also drags in a third
    cached region read — the width adjoint's own file, whose region monitor
    records the FULL λ grid even when the FOM uses one λ. Port-only fits
    160 G at 501 points; add the width entry and it does not.
    RULES: (a) a wg_pure gate (J = −softW) may run a COARSE λ grid — the T
    spectrum enters that FOM nowhere, so cut n_wl_points (keep it ODD so the
    centre λ stays on-grid); (b) any campaign with a second fom entry gets
    250-300 G, not the 160 G habit; (c) `validate_gradient` uses CENTRAL
    differences = 2 forwards PER INDEX — size the lane as
    fwd + adj + 2·n_indices forwards (3 indices ≈ 6 h here) and never put it
    in the 2 h lane. Both failed attempts at this gate (136108 TIMEOUT,
    136122 OOM) were LANE-SIZING errors, not physics errors — the physics
    (GPU import-source adjoint, 3,135 s) passed both times.

27. **★★★THE rho DEADBAND WAS THE HOLE — ALWAYS CONVERT A CONSTRAINT INTO THE
    SPEC'S OWN UNITS BEFORE TRUSTING IT (measured 2026-08-18, job 134217).**
    The audit's 3 rows form a clean 2-factor factorial:
        origin  rho 1.0000 2Ss   0.0 | FWHM 17.100  sigma 17.487
        noshift rho 0.9722 2Ss   0.0 | FWHM 19.165  sigma 17.503
        best    rho 0.9722 2Ss 130.6 | FWHM 21.709  sigma 17.800
    - corrugation alone: **FWHM +12.1%, sigma +0.09%** (sigma ~28x less
      rho-sensitive than FWHM — effectively BLIND to apodization)
    - shifts alone:      **FWHM +13.3%, sigma +1.7%**  (~4.8x less sensitive)
    ★ROOT CAUSE: `RHO_DN = 0.95` let rho fall 5%, and
    5% x (74.3 um/17.1 um) = **+21.7% FWHM**. The constraint written to PROTECT
    the mode width PERMITTED a fifth of width growth, and sigma's +2% band
    rubber-stamped it because sigma cannot see rho. Nobody had ever expressed
    the rho band in microns.
    ★THE RULE: a constraint stated in a surrogate's units (rho, a moment, a
    ratio) is meaningless until you MEASURE its conversion to the spec's units
    and check the implied slack. Do that at design time, not after a campaign.
    ★SHIPPED: `FWHM_A_RHO -74.3` um/unit, `FWHM_A_SHIFT +0.01948` um/nm,
    `FWHM_RESID_WARN 0.30` — GRADIENT HINT ONLY (rho is the mean of a tapered
    profile => shape-specific, item-24 class). Authority = the MEASURED
    `fwhm_env_um` guard.
    ★COROLLARY MEASURED THE SAME DAY: **no grating-side T gain has ever been
    demonstrated at constant FWHM.** The comb IS honest — bare-uniform T
    0.88073 vs comb-uniform 0.89265 at identical knobs = +0.0119 at fixed
    width (reproduces the A0 gate's +0.0105). Apodization near the origin is
    ~2x more width-efficient than the optimizer's converged moves (seedB ev1:
    0.0075 T per %width vs 0.0035), so a properly fenced campaign is expected
    to land well below the sigma-era 0.964 headline — measure, don't assume.

## ═══ THE GENERAL METHOD (v2 era, 2026-08-22) — for ANY new constrained
## inverse-design problem the user brings. Device-independent; every rule
## below was paid for by a measured incident in THIS program.

1. **Spec observable first.** Fix the exact measurement convention (ONE
   function, shared with the trusted post-processing, golden-file tested
   against stored values) BEFORE any optimization exists. Our y-integral bug
   voided weeks because the observable itself was broken.
2. **Never control a proxy without co-measuring the spec, every eval.** Log
   proxy AND spec per evaluation with a loud divergence alarm (item 25).
   Convert every constraint band into SPEC units at design time (item 27).
3. **Match the surrogate CLASS to the observable class.** Moment-type
   surrogates are structurally blind to level-set observables (σ missed
   +26.6% FWHM as +2.5%; participation ratio equally blind). Level-set spec
   ⇒ soft level-set surrogate (softW pattern: smooth → soft-threshold →
   integrate; validated ≤2 pp where moments err 24 pp).
4. **Enumerate cheat channels at design time**: any linear combination of
   free knobs that reconstructs an excluded knob (Σshift = cavity length)
   gets a differentiable wall. Guards fire on ACCEPTED-BEST only, never on
   probes. Restart selection filters on measured-spec compliance.
5. **Constraint architecture, in order of preference:** (a) MEASURE the
   cheapest monotone payback knob early (ours: uniform corr-add costs
   0.002 T/µm — 20× cheaper than assumed). If payback ≪ the objective's
   spec-efficiency, PROJECTION-FIRST wins: optimize the objective, re-trim
   on the measured spec at stage boundaries. (b) AL penalty on a
   delta-anchored surrogate, multipliers updated on MEASURED violations,
   filter acceptance. (c) In-loop constraint adjoint only if (a)+(b)
   thrash — and price it first (our width-adjoint: GPU-unsupported, ≥5 h
   CPU — nearly priced out by (a)).
6. **Gate ladder classes (transferable):** G0 = math (autograd-vs-FD of the
   fct; surrogate-vs-spec tracking on STORED data — zero GPU); G1 = local
   build + generate() + adjoint-setup property smoke (each cluster failure
   class becomes a new local gate); G2 = forward canary vs anchors; G3 =
   gradient FD gate with a per-adjoint-path C calibration (NEVER assume a
   solver's adjoint normalization — every path gets its own measured C);
   then CAMPAIGN-AS-GATE: dispatch and judge the first ~3 evals (user
   calibration 2026-08-22: improvement IS validation once guards make
   cheating impossible; failure costs hours, not correctness).
7. **Seeds must be measured-in-spec at start** (the dip seed was +2.5% over
   band AT BIRTH and nobody knew for weeks).
8. **Build a zero-GPU calibrated ranker before spending GPU**: fit a cheap
   physics model to stored measurements (ours: light-cone integral, rank
   corr 0.975, slope 0.32 = compressive). Use it to CLOSE directions
   (chirp, sinc) and RANK candidates (Gaussian core) — never to predict.
9. **Structural counts:** measured-flat ⇒ freeze; genuinely uncertain ⇒
   give it an exploration mechanism (never a silent freeze).
10. **Reporting:** only re-trimmed, equal-spec, production-convention
    numbers are results. Never compare across spec values, meshers, or
    windows. Everything else is "candidate".
11. **Ops (unchanged, §6/CLAUDE.md):** resume >2 h, one array per decision
    point, fetch-early, in-study anchors, seat probe, per-study lists.

31. **★THE DISK JANITOR NEVER WORKED — VERIFY A CLEANER DELETES, DON'T TRUST
    THAT IT RUNS (root-caused 2026-08-23).** `h5_roll_clean.sh` grouped files
    by the h5's OWN parent directory (`find … -printf '%h'`), but lumopt2
    writes every `*_output.h5` into its own subdirectory
    (`<label>_files/fwd_default_iter0/fwd_default_iter0_output.h5`). So
    "keep the newest 2 per directory" always ran `tail -n +3` on a ONE-file
    list and removed nothing — for weeks. Every quota event traces to it
    (job 136090 killed at the 330 G hard limit; two further near-misses at
    271 G and 287 G, each "fixed" by hand-deleting, which masked the bug).
    FIXES, both needed: (a) group by the enclosing `*_files` directory so the
    newest two — this iteration's forward + adjoint — survive and older
    iterations are reaped; (b) run it from **cron** (`*/10`), never as a
    login-node `nohup` daemon: it died with the session three times in one
    day, and a dead janitor is invisible until the quota bites.
    GENERAL RULE: a janitor is not "working" because the process exists —
    verify it has actually deleted something (watch the quota fall, or dry-run
    its selection). Pair with item 29: never key it to a study-name glob.
    Steady-state arithmetic worth knowing: each concurrent campaign holds
    ~12 GB of live scratch (fwd + adj ≈ 5.9 GB each), so four campaigns need
    ~50 GB of headroom ON TOP of baseline occupancy.

32. **★★THE WIDTH WALL'S SLOPES WERE BOTH SECANTS — MEASURE THE CURVE, DON'T
    FIT A PAIR (2026-08-23/24).** `FW_A_ELONG = 0.01355 um/nm` was the secant
    of ONE pair (fspw_noshift -> fspw_best, elong 0 -> 130.6) applied as a
    local slope everywhere. Measured truth (IGUM 61742 + 61782, 6 rungs on the
    uniform corr-325 seed, pure common mode, pitch-locked mesh):

    | e = 2*sum(shift) nm | 0 | 60 | 120 | 180 | 240 | 287.5 |
    |---|---|---|---|---|---|---|
    | fwhm_env um | 18.345 | 18.311 | 20.483 | 24.015 | 28.768 | 32.698 |

    A THRESHOLD: flat to ~65 nm (e=60 measures NARROWER than the seed), then a
    knee and a steep, still-accelerating rise. `dW = 7.8654e-3*max(0,e-65)^1.39`
    fits all six to 0.106 um vs the 0.367 um half-band. Engine:
    `_fw_elong_curve` + spec flag `fw_curve` (default False). The interim
    quadratic `fw_convex` (FW_C_ELONG) is ALSO refuted — do not enable it.
    CONSEQUENCE: the old wall charged +0.813 um of predicted widening at e=60
    where the true cost is ZERO — a penalty ~0.795 against a whole FOM of
    ~0.67. Campaign 136466 was thereby FORBIDDEN from its own subject: it
    oscillated e = 0 -> 287 -> 0.3 -> 144 -> 0.7, never probing 1-100 nm, and
    gained +0.0005 T in 7 h while shift-FROZEN 136468 gained +0.0076.
    `FW_A_MCORR = -0.0470` is the same failure class: it is the full-range
    secant of the 9-row retrim curve (mcorr 315.97 -> 375.97, verified), whose
    LOCAL slopes run -0.044 to -0.029, and campaign 136468 measured -0.0666 at
    mcorr 295 — outside the fitted range entirely. GENERAL RULE: any steering
    slope in the FOM must record the RANGE it was fitted over, and a campaign
    that operates outside that range is running on an extrapolation.

33. **★OPERATIONAL TRAPS FROM THE 2026-08-23/24 RESTART (each cost real time).**
    (a) RESULTS PATH: outputs live at `results/<study>/results/<label>/…` — a
    glob on `results/<label>*` matches the STUDY directory and silently finds
    nothing. This produced a false "tasks failed" alarm when the tasks had in
    fact exited 0.
    (b) DEPLOY + PERMISSION CLASSIFIER: the compound form
    `cd … && ENV=… bash deploy | grep | head` was BLOCKED; the plain
    `ENV=… bash athena/deploy_athena.sh --lumopt2-design=…` (no cd, no pipes)
    went through. The block landed AFTER the campaign had already been
    scancelled, briefly stranding an empty slot — order a restart so the
    cancel happens only once the dispatch path is known clear.
    (c) MONITOR STALENESS: a monitor keyed to a campaign LABEL keeps reporting
    the dead log after a relabel (s2 -> s3) and you go blind to the live one.
    Re-point the monitor in the SAME turn as the restart. And never put raw
    resource numbers (quota GB, seat counts) in the change key — band them, or
    every janitor sweep costs a model turn.
    (d) WALLTIME SIZING: size it from JOB START to row, not from the solve
    time. The steady-state forward solve is ~33 min, but project setup pushes
    the first row to ~2 h on Athena; a 1:30 walltime sized off "33 min" left
    far less margin than intended.
    (e) IGUM LICENSE RACE: a task cold-starting on a node where sibling tasks
    just finished can lose the ansyscl checkout ("ANSYSLI exited or could not
    read server port ansyscl.<node>…"). Casualty is cheap — resubmit that one
    index with `--array-tasks=<i>` after the queue drains.

34. ★RANK-DEFICIENT-SURROGATE trap (2026-08-24, Fable audit; the reason
    2-param hand rules beat the 51-param optimizer): any penalty built on a
    SCALAR summary of a param block (mean corr, total elongation) gives
    L-BFGS-B an identical gradient across that block — every direction that
    redistributes within the block is unpriced, and the optimizer converges
    to the wrong fixed point (not slowly to the right one). Measured: wall
    said −0.82 µm for the see-saw move, truth −0.015 µm. Fix pattern =
    per-parameter measured weights (`fw_tooth_w`/`FW_TOOTH_W`, anchor gains
    `corr_vec`). Rule: when a hand-designed move beats the optimizer, check
    FIRST whether that move lies in a surrogate's null space. Corollary: an
    unpriced free channel (wcav) is the same hole at rank 0 — the measured
    guard owns it, but list such channels explicitly in the campaign
    docstring. PSO is not the answer to "optimizer stuck" here (~85 min/eval
    kills population methods; the gradient was fine — the prices were wrong).

35. ★★ITEM 24 REPEATED ON A NEW SURROGATE — RE-ANCHORING IS NOT
    RE-FITTING (2026-08-24). Item 24 taught TWO things about the sigma-hat
    wall: (a) re-anchor at measurement cadence, (b) the surrogate DOES NOT
    TRANSFER BETWEEN BASINS. When sigma was retired and `fwhm_wall` built to
    replace it, only (a) was carried over. Its constants (FW_A_MCORR,
    FW_CURVE_C, FW_TOOTH_W) are all fitted on the UNIFORM corr-325 device and
    then applied to apodized ones. MEASURED consequence: on BEST_T9636
    (mcorr 357.95) the elongation curve predicts 2.748 um of widening for
    e=132.6 where the truth is 1.4994 um — over-taxing by 1.83x, on a
    campaign whose entire purpose is exploring elongation.
    WHY RE-ANCHORING DOES NOT SAVE YOU: the anchor is an OFFSET. It pins
    fhat to a measured width at the current point, so the model is exact AT
    the anchor and wrong as soon as you step — with an error set by the
    SLOPE, which no amount of re-anchoring touches. A delta-anchored wall
    with a wrong slope is a correct value and a wrong gradient, and the
    gradient is the only part the optimizer uses.
    RULE: every surrogate constant carries the device class it was fitted on.
    Before reusing a width/coupling constant on a device from a different
    class (uniform vs apodized vs shifted), either re-measure it there (2-3
    forwards) or state the transfer as an EXPECTED assumption in the runner
    docstring. Physical reason it cannot transfer, from the user
    (2026-08-24): once the device is not uniform the envelope is no longer a
    single exponential -- there is no one kappa to put in the exponent, so no
    one constant describes the decay. Corollary for the FIX itself: the
    3-block FW_TOOTH_W is fitted on the uniform seed too, so it inherits this
    caveat -- it is right about ORDERING (inner teeth cost ~10x outer, which
    is what the rank-1 wall got wrong) and provisional about MAGNITUDE.

36. ★THE JANITOR DIES — CHECK IT EVERY POLL, NOT ONCE (2026-08-24, twice in
    one day). `~/h5_roll_clean.sh` was found DEAD at the start of the session
    (quota 267/300 G with ~3.5 h of runway), restarted with plain
    `nohup ... &` — and was dead AGAIN ~2 h later, during which quota climbed
    214 → 235 G. Plain nohup from an ssh session does not reliably survive on
    the Athena login node. Restart it DETACHED:
      `nohup setsid ~/h5_roll_clean.sh >> ~/h5_roll_clean.log 2>&1 < /dev/null &`
    and put `pgrep -c -f h5_roll_clean.sh` in the SAME ssh as every status
    poll — it costs nothing and this is the failure that silently hangs jobs
    at container init. Item 31 said "verify a cleaner deletes, don't trust
    it"; the 2026-08-24 addendum is "verify it is still ALIVE, every time" —
    a cleaner that ran once is not a cleaner that is running.
    ALSO measured this day: the janitor caps GROWTH but cannot reclaim dead
    studies (it keeps newest-2 per `*_files` dir, and each finished study has
    only 1-2). Reclaiming ~85 GB of cancelled-campaign `*_output.h5` needed
    an explicit purge (283 → 203 G) — ask the user, it is a deletion.
    ★2026-08-27: the FIXED once-per-10-min cron cleaner (`athena/h5_clean_once.sh`,
    adds PASS 2 for dirs cold >24 h) is INSTALLED on Athena, md5-verified.
    It is a CRON job — check `quota -s` + the h5 total, never pgrep.

37. ★DEFECT #19 + THE λ-CHAIN (2026-08-25/27) — THE WIDTH GRADIENT WAS THE
    WRONG DERIVATIVE. gW from the width adjoint is ∂W/∂p at FIXED λ, but the
    spec width lives at the device's own MOVING resonance, and W is slaved to
    λ (dW/dλ ≈ +0.3655 µm/nm uniform / +0.300 seesaw — per-run, ~20% spread,
    NOT a constant of nature; re-derive per seed family via
    `gates/derive_dwdlam.py`). 93%/77% of the width blow-up that killed both
    baselines was resonance drift. FIX: gλ = dλ_pk/dp from the IFT on
    ∂T/∂λ=0 via a MATCHED antisymmetric stencil pair (exact for any h on a
    symmetric lineshape; the naive pair errs 1/(1+x²) = 49.4% low at x≈1),
    two selector passes off the same solved fields = ZERO extra adjoints;
    gW += wg_dwdlam·gλ. Guards: dTp<0 (straddles a max, else LOUD skip),
    1<i_pk<len-2 (edge wrap), ≥40 spectrum pts per spectral FWHM. FIVE
    offline gates before any dispatch, each with an expected last line
    (HANDOFF top box): lam_chain math + plumbing + projection + predispatch
    + derive_dwdlam. `validate_c325` task 41 = the 3-iterate hardware toy
    (fresh label, cold start); task 27 = its wg_lam_chain=False CONTROL TWIN
    under the same engine/mesh. Job 137845 (2026-08-27) is the first hardware
    run of both. Until it completes cleanly the λ-chain is UNVALIDATED.

38. ★MODEL-DELEGATION WORKFLOW (user directive 2026-08-27, after a token
    audit found 96% of 2 weeks' burn in two marathon sessions and 13% in
    hand-rolled queue polling). FABLE IS THE DECISION MAKER: planning,
    gradient math, verdicts, anomaly root-causing, dispatch go/no-go.
    OPUS SUBAGENTS (Agent tool, model:"opus", background) execute routine:
    monitoring/polling, result fetch + MATLAB plotting, log summarization,
    jsonl data crunching (Fable reviews conclusions), skill/memory drafts,
    quota/seat probes. Standing burn rules distilled from the audit:
    (a) queue-watching goes through the Monitor tool or one background
    watcher — never repeated squeue turns in the main loop; (b) one ssh per
    poll folding squeue+sacct+log-tail+quota; (c) the parent never Reads a
    subagent's tasks/*.output transcript — the report IS the interface;
    (d) state docs get ONE batched edit per stage, not incremental edits;
    (e) read big state files (HANDOFF 200+ KB) by top box / Grep section,
    never whole; (f) cap sessions — invoke safe-compact on a turn budget.
    Once the corrected pipeline is routine, plain-Opus sessions carry it and
    Fable is reserved for new failures / math / physics contradictions /
    >1-GPU-day decisions.

40. ★PERIODIC DEEP-CHECK (user rule 2026-08-28: "keep asking what's going
    on" — pattern-greps only catch what they were told to catch). While any
    hardware run is active, every ~2 h an OPUS background agent (never
    Fable) does ONE folded ssh + local jsonl read and answers four
    questions: (1) is progress at the expected cadence (solve times vs
    history — a stall looks like silence to every grep); (2) any log
    anomaly OUTSIDE the standing grep set (read the last ~100 lines with
    fresh eyes); (3) quota + janitor state; (4) are the small state files
    (jsonl) pulled local (no unique data on the cluster). One-paragraph
    report; wake Fable only on anomaly. Cadence deliberately ~2 h, not
    more: each poll spends the ssh connection budget, and the marginal
    value of quiet checks decays fast — event monitors carry the
    minute-scale layer. Idle periods (nothing running): once per session
    is enough.

39. ★PIPELINE SMOKE TIER (user rule 2026-08-28, after the analysis-mode
    crash cost a full dispatch cycle at hour 1 of an 11 h run). The debug
    ladder for engine changes is now THREE tiers, each mandatory before the
    next: (1) offline gates (seconds, math + call path + source-structure);
    (2) `validate_c325` task 35 pipeline smoke (~1.5-2 h GPU: the SAME
    191-param spec and code paths on an N=40 low-Q surrogate — the only tier
    that catches LIVE-SESSION-STATE bugs like the in-analysis-mode dEps);
    (3) the physics run (toy/campaign). Local gates provably cannot see
    session-state hazards — five of them passed while 137845_41 died. Never
    quote smoke numbers as physics. Related trap the same night: task
    indices must be checked against MEMBERSHIP branches (_GFR_RUNGS ate
    index 27), not just literal `== n` matches.

41. ★JANITOR CRONS ARE PART OF THE NUMERICS ENVELOPE (2026-08-29, killed
    campaign c1 twice). Any cleanup automation must satisfy TWO bounds
    proven against the SLOWEST live consumer, not the typical one:
    (a) age floor > the longest window a running gradient still needs a
    file (a preemption-resumed iterate held its forward h5 live ~2.5 h —
    the 30 min floor deleted it mid-gradient, signature
    `Can not find result 'E' in field_profile_adj` 40-50 min into
    assembly); (b) keep-count > the number of files simultaneously live
    (forward + port adjoint + width adjoint = 3; "keep 2" ranked the
    forward third and killed it). Current safe values in
    athena/h5_clean_once.sh: -mmin +240, keep 4. Corollaries: a crash that
    appears ONLY after preemption/requeue may be an interaction with
    time-based automation, not the resume code — check cron/mtime
    coincidence FIRST (the fwd_default/ dir mtime matching a */10 cron
    firing was the confirming fingerprint, free); and every watcher's grep
    set must include `Can not find result` (it is the license-no-op AND
    the deleted-scratch signature).
