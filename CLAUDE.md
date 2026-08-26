# CLAUDE.md — Pi-Shift Bragg Grating FDTD

Project rules for Claude Code. These are always-on invariants. They were distilled
from ~35 prior sessions; the incidents behind each rule are real and cost real GPU
hours. Read `README.md` for architecture and `runners/README.md` for the study patterns.

The device is a **pi-shift Bragg grating** (use this term in discussion/writeups).

> ## ★★★CURRENT PROGRAM STATE — READ BEFORE ANY INVERSE-DESIGN WORK
> **`runners/lumopt2_design/HANDOFF.md`** is the live, self-contained state of the
> lumopt2 inverse-design programme (2026-08-18). Read it before touching that
> programme, quoting any of its numbers, or resuming a campaign.
> **`runners/lumopt2_design/THEORY.md`** is its companion: the METHOD, not the
> state — what the cost function is, the 191-parameter layout, how the two width
> measures relate, the projected-gradient algorithm, how the adjoint gradients
> are obtained (tiling, C_field, the zero-extra-solve split), the resonance
> chain-rule term, and the best design we hold. **Read THEORY.md before
> reasoning about the optimizer or the gradients; read HANDOFF.md before
> running anything.**
> **The one fact that changes how you read everything else:** the engine's mode-
> profile extraction never integrated over y, so **every `sigma` and `FWHM`
> logged before 2026-08-18 is VOID** (T / λ / Q / R / loss are unaffected). Mode
> width is measured ONE way only — `sim_helpers.extract_and_process_field_profile`,
> the same convention as `post_processing`'s `fwhm_m`. A raw-line variant, fitted
> width slopes, and a coupled-mode-theory model were all tried, all wrong, and all
> deleted by user order; do not reintroduce them.

---

## 1. Where things run

- **Both clusters work; ASK which one before dispatching** (user rule 2026-08-07):
  a plain one-line question ("Athena or IGUM?") — unless the user already named the
  cluster for this task, in which case just do what they said. (The older
  "Athena-by-default, don't ask" convention is superseded by this rule.)
- Athena dispatch: `bash athena/deploy_athena.sh`.
- **Athena runs Lumerical from a container** (`~/containers/lumerical-2026R1.sif`,
  filename fixed — ~6 job scripts hardcode it; engine inside = **2026 R1.3 build
  4572 since 2026-08-12**, matching IGUM and the local Windows install). To put a
  new Lumerical version in it, use the **`update-container` skill** — the 5 GB
  `.sif` never crosses the VPN, and never delete an old version's artifacts (old
  sifs are renamed: `lumerical-2026R1.2.sif`, `lumerical-2026R1.1.sif`, plus the
  parked `~/lum_r1*_parked_*` trees). An engine bump is a §2 named-numerics
  change: it ends with a canary vs a stored control, on **both** clusters.
- **IGUM (ECE faculty cluster) is a second, coexisting option** — `bash
  igum/deploy_igum.sh`. Native Lumerical — **containers are impossible there**
  (no apptainer/singularity, docker daemon denied; verified 2026-08-12), so a
  version bump means an extracted RPM tree we own:
  `~/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261` is the live
  `LUM_HOME` (the admins' `/apps/ansys/Lumerical-2026-R1.2` stays as fallback).
  IGUM has no `rpm2cpio` — extract on Athena, tar-stream over the LAN. Submission needs
  `--account`+matching QOS, `part-preempt` is preemptible (sweeps OK, long stateful
  optimizations stay on Athena), and **license seats are SHARED with Athena** —
  probe both before big runs. Athena stays the default. See `igum/README.md`.
- **Local is allowed** for: building scenes, `save_fsp`, smoke tests, MATLAB plotting,
  and any quick non-GPU check. Local `fdtd.run()` is slow — only do a real local FDTD
  run if the user explicitly asks.
- GPU/partition: **just use the default** (don't ask). The one exception worth a
  one-line heads-up: a long, stateful optimization on a `*-shared` partition can be
  preempted and lose progress — if that matters, mention `--gpu=a100` (non-preemptible)
  but don't block on it.

## 2. Resonance & metrics (correctness-critical)

- **Always use the built-in resonance finder.** Never pick resonance by `max(T)` /
  `argmax(T)` — the global T max sits in the passband (~1570 nm), not the defect peak.
  Use the stored `resonance_wavelength_nm` field, or `plot_transmission.m`'s peak finder.
- **"FWHM" means spectral FWHM** (`spectral_fwhm_nm`, from T(λ)) unless the user says
  "spatial". `spectral_fwhm_nm` is often stored **negative** → use `|spectral_fwhm_nm|`.
- **Q = resonance_wavelength_nm / |spectral_fwhm_nm|.** When the user asks for "the
  wavelength," they usually mean the resonance wavelength (for Q). `fwhm_m` is the
  *spatial* mode width (energy vs x) — used for corrugation/mode-width matching, not Q.
- **Post-run sanity check before trusting/continuing on any FDTD result:**
  1. `resonance_wavelength_nm` exists, is finite, and lies inside the scan window.
  2. Peak T is above a sane floor (dead device shows T≈0.0008). TM healthy peaks can
     still be ~0.83, so use a low floor, not a TE-tuned one.
  3. If either fails: **stop and surface it** ("no resonance found / off-window / dead
     device") — do not silently build downstream conclusions on it. A "converged"
     optimization on a dead device returns confident garbage.
- **Single-λ monitors/extractions key off `resonance_wavelength_nm`** (its index in the
  recorded band) — never "1 frequency point + use source limits", which records at the
  band-center *frequency* (≈1546.4 nm here), not the resonance. Far-field got plotted at
  the wrong λ twice this way.
- **Absolute T/loss are numerics-sensitive; compare only within identical numerics.**
  For strongly-radiating variants (e.g. TM corr-400, 16–19% resonant loss) the
  transverse box size alone moves absolute T by ~3 points (3.8→4.8 µm: 0.828→0.799),
  and mesh mode moves it again — the old "1.8λ span changes T negligibly" claim does
  NOT hold there. Every sweep must carry its own in-study no-change control at the
  exact same numerics, and all reported Δ's are vs that control. If the absolute
  matters (fab comparison), run a domain-size convergence check first.
- **A candidate effect near the numerical noise floor is not a result.** Measure the
  floor inside the sweep (repeat a few points offset by half a mesh cell) and confirm
  survivors at `simulation_mode="accurate"` before claiming them (2026-07-02: pillar
  +0.0020 T sat exactly at the dx=50 nm jitter floor 0.0018; at dx≈35 nm the jitter
  collapsed to 0.0001 and the effect survived — that two-step is the template).

## 3. Mesh / accuracy

- **`simulation_mode = "optimization"`** (dx=50nm) is the default and the right choice
  for sweeps and optimizations.
- `"accurate"` (dx≈35nm) is reserved for **final / fab-comparison validation** — it is
  case-dependent, not automatic. Don't switch to it without reason.

## 4. Geometry & materials (defaults — confirm before new TM work)

- **Indices are stable:** `n_core = 1.97`, `n_clad = 1.444`.
- **TM anchored geometry is per-height and is a DEFAULT, not a constant** — it gets
  changed in many places over time. For **height 350 nm**: pitch **516.83 nm**,
  corrugation **400 nm** (co-resonant with TE + width-matched). pitch ↔ corrugation are
  **coupled** (change one → re-trim the other). Other heights use a different pitch the
  user supplies. **At the start of any new TM task, confirm height + pitch + corrugation**
  rather than assuming these defaults.
- When material index or pitch changes mid-study, **re-scan the baseline** at the new
  resonance (don't reuse the old scan window — that's how peaks get missed).
- **"N periods" means `n_periods_each_side`.** Baselines: TE = 80/side; TM
  period-matched to TE@80 = 132/side.
- **Pitch-retune acceptance default:** present the residual detuning Δλ and accept when
  it is ≲1 nm, unless the user objects (asked twice, user accepted 0.75 nm).
- **Before dispatching a new scan, state the target resonance λ and scan-window width**
  in one line and sanity-check them against the study (past incidents: a 75 nm window
  where ~20 nm was meant; aiming at 1449 nm when the user meant 1550). Don't block on
  it — but if they conflict with something the user said, ask first.

## 5. Verification policy (smoke-test, don't over-test)

History is one-sided: under-testing repeatedly burned GPU hours (dead parametric TM
device ~8 GPU-h; bad-gradient lumopt ~30 GPU-h; `phi=-90` source wasted weeks).
Over-testing never once cost anything. So:

- **Smoke-test before dispatch when the change touches:** (1) device geometry, (2) a new
  builder / parametric scaffold, (3) inverse-design / gradient equations, (4) source or
  boundary-condition setup. Especially for anything **new**.
  - Lumapi: local build-only `save_fsp` (<1 min) + eyeball the geometry.
  - **Config-override trap (burned 2026-08-13):** SimulationConfig dataclasses accept
    UNKNOWN attributes silently — `cfg.grating.corrugation_depth_m = ...` creates a
    dead attribute (corrugation lives on `cfg.geometry.*`) and the device builds at
    the default. After any direct-attribute override, verify the built values via
    `SPEC.expand()` / `describe()` / a build printout before dispatch.
  - **Any edit to `bragg_device.py` geometry/monitor code:** run
    `python debug_fsp_compare/scene_snapshot.py --out <tmp>` and diff against
    the committed `debug_fsp_compare/snapshots/` references (6 configs spanning
    the builder's code paths; byte-identical = behavior preserved). Regenerate
    the references only when a geometry change is INTENDED, and say so.
  - Parametric/PSO builders: score gen-0 / seed against the known-good baseline; if it
    doesn't match, the builder is broken (use `rebuild_per_particle`).
  - New gradient method: finite-difference `check_gradient` on a tiny problem before
    scaling (hard gate: `vec_error` must be small).
  - **★A MATH GATE IS NOT A PLUMBING GATE — smoke the CALL PATH too (2026-08-26,
    2 GPU-h).** Any new fct / jacobian / adjoint-assembly code must be driven
    through the REAL wrapper (build the actual fct, call `autograd.jacobian` on
    it) locally before dispatch, not merely verified as a formula. Job 137267
    died at 2:03 on `IndexError: invalid index to scalar variable` while its
    math gate passed at 0.0034%: the fct's `x` is the **FLAT** vector
    `[T(λ_0)…T(λ_n), softW]`, NOT a list of FOM entry results — which is why
    `x[-1]` is the width. A <1 s local gate
    (`runners/lumopt2_design/gates/gate_lam_chain_plumbing.py`, asserting a
    one-hot jacobian AND that the old broken form still raises) catches it.
    That `gates/` dir also holds the math, projection and bounds gates — run
    all four before any lumopt2 dispatch. Corollary: a gate that cannot fail proves nothing — assert the
    known-bad form still errors. Second corollary from the same fix: count how
    many FIELD SETS are live at once before adding an assembly pass (the
    double-pass already OOM-killed a 160G job at 501 λ) — convert each to its
    parameter vector and free it rather than stashing field sets.
  - **Designed recovery paths get an END-TO-END smoke through the real wrapper
    stack** (2026-08-16: both campaigns died because a guard exception was
    tested at its raise site but lumopt2 double-wraps exceptions —
    scipy_optimizer.py:583 without `from e`, optimization.py:852 with — and
    the catch never matched; walk BOTH `__cause__` and `__context__`).
    Replicate the third-party raise chain locally and assert the handler
    engages before trusting any except-and-recover design.
  - MATLAB: `checkcode` lint + headless `exportgraphics` render.
- **★DEBUG ON THE SMALLEST SCENE THAT CAN ANSWER THE QUESTION — never on the device
  (user rule 2026-08-24, after a night of it).** Before dispatching a diagnostic, ask
  what the question actually depends on. A question about **numerics, an API, a solver
  limit, a crash signature, or a launch/config error does NOT depend on our grating** —
  it needs an empty box, a dummy source, a short sim time, and it answers in SECONDS.
  Only questions about the DEVICE PHYSICS (T, λ, Q, mode width, gradients of those)
  need the real device, and even then prefer the smallest N that keeps the physics.
  INCIDENT: the FieldRegion-on-GPU `invalid configuration argument` was chased with
  FULL-DEVICE rungs at 45-70 min each across four jobs (136799/136826/136869/136907),
  making every bisection step cost an hour — for a CUDA kernel-launch bound that has
  nothing to do with the grating. `runners/lumopt2_design/gpu_probe.py` answers the
  same question over 12 sizes in one short job. Cost of the lesson: ~6 GPU-h and most
  of an evening.
  Corollaries: (a) bisect a threshold in ONE array of cheap tasks, never one
  expensive point per dispatch; (b) anything checkable with a build-only `save_fsp`
  or a local dataset/shape assertion must be checked that way FIRST (zero GPU);
  (c) this is the same principle as the existing `check_gradient`-on-a-tiny-problem
  gate below — apply it to solver/API questions too, not just gradients.
- **★VALIDATE THE PARAMETER VECTOR AGAINST ITS OWN BOUNDS BEFORE EVERY DISPATCH
  (2026-08-25 — this class cost FOUR dispatches in one night).** lumopt2 rejects an
  out-of-bounds seed outright (`parametrization.py:674 _check_params`), and the job
  dies in ~60 s having queued behind everything else. The trap is always the same
  shape: a spec that FREEZES something (e.g. `free_comb=False` ⇒ comb bounds collapse
  to ±0.001 nm) combined with a seed or a DETUNE point that moves it — `BEST_T9636`
  carries comb r = 80.1386, and `run_adjoint_only`'s detune=1 sets the centre post to
  100.0. Reproduce the runner's exact vector locally (seed → detune → clamp) and check
  it against `param_bounds(spec)`; it is a two-second numpy check with zero GPU.
  Reusable checker: `runners/lumopt2_design/gates/predispatch_check.py`. Corollary: when a fit or gate
  must sit at the SAME operating point as a stored reference, the spec must ALLOW that
  point — freeing the comb changes only the bounds, not the geometry at an explicitly
  set point.
- **Skip** re-verifying known-good baselines and re-linting untouched code. Don't invent
  extra test passes for mechanical edits.
- **All local verification runs are SILENT** (user rule 2026-08-07): lumapi always
  `hide=True` (set in `bragg_device`; pass it in ad-hoc scripts too), MATLAB always
  `-batch`. Nothing opens a window on the user's screen during automatic
  build/smoke/plot steps.

## 6. Server safety

- **A run is never a trivial action — think first, run second** (user rule 2026-08-07,
  after the flush-ladder mesh artifact burned ~20 GPU-h). Before ANY dispatch or long
  run: state what the run will decide and why existing results can't answer it; prefer
  the smallest discriminating experiment. After ANY anomaly (unexpected λ/T/fwhm,
  <30 s crash, off-family value): NO new runs until the cause is understood via free
  diagnostics first (stored .mat comparisons, scene diffs, job/solver logs, local
  build-only rebuilds). Runs must be consistent with the program's existing
  measurements — a run at silently different effective numerics (e.g. a changed mesh)
  is worse than no run.
- **Never re-measure a stored result — CONTROLS above all** (user rule 2026-07-26,
  hardened 2026-08-10: "if we have a result somewhere don't do again — very
  important"). Before any dispatch, enumerate which requested points already exist
  (results_from_athena/, results_from_igum/, memory) and cut them; the dispatch note
  says "point X reused from <job/file>". Default = NO control row — cite the stored
  baseline file. Cross-cluster reproducibility is PROVEN (2026-08-10: Athena
  corr-325 N165 ctrl T 0.4906 / Q 13930 ≡ IGUM-stored, exact), so a cluster switch
  alone does NOT justify a control re-run. The only valid justification is a NAMED
  §2 numerics change (box, window/points, mesh, symmetry/BCs) vs every stored
  baseline, written in the runner docstring. A stored identical-numerics control
  satisfies §2's in-study-control requirement.
- **★Never let one cluster hold UNIQUE results — fetch early (2026-08-17).**
  A long campaign's incremental log (eval jsonl / params history) is unique
  data the moment it is written; IGUM went unreachable for hours holding the
  only copy of seedB's best geometry. Rule: pull the small state files
  (jsonl/csv, ~KB) on every milestone check, not at study end — cost is
  seconds, and CLAUDE.md §6's "reduce field data server-side" concerns the
  BIG .mat/field volumes, never these. Cluster-choice corollary (measured
  this program): both clusters earn their keep via PARALLEL throughput
  (two seeds in one night = the convergence evidence), and IGUM adds
  no-preemption; but IGUM's INFRASTRUCTURE is the weak link (slurmdbd down,
  login flaps, hand-maintained Lumerical tree) while its COMPUTE is fine —
  so give IGUM long self-contained resume-protected runs, keep interactive
  / closely-monitored / fast-iterating work on Athena.
- **★Login-node connection budget (burned 2026-08-17): ≤~3-6 ssh/hour per
  cluster for automated polling, ONE connection per poll** (fold lmstat/log/
  queue probes into the same ssh, never open a second). IGUM began refusing
  our key ~80 min after a monitor polled it 24×/h; ~45 min of zero contact
  restored it. On ANY auth refusal ("Permission denied" with port 22 open):
  STOP all automated contact ≥45 min, then ONE probe — never retry-loop
  (retries deepen rate-limit bans, and IGUM's sshd also flaps on its own —
  refusal ≠ proof of ban). Cluster JOBS are unaffected by login-node auth
  (compute-side, afterok chains still fire) — an outage costs visibility,
  not science, so never panic-redispatch because the login node is refusing.
- **ssh/scp command form.** Always write remote commands host-first:
  `ssh evyatarrubin@athena.technion.ac.il "..."`. Never env-var-prefixed forms
  (`SSHHOST=... ssh "$SSHHOST" ...`) — they evade the permission-rule pattern matching
  (including the `scancel` ask-guard). Strip the Technion login banner with
  `grep -vE "post-quantum|openssh|may need to be upgraded"`.
- **Concurrency / no clobbering.** Deploy does `rsync --delete` into a *shared*
  `REMOTE_BASE/project/` and writes to a *shared* `results/` + `data/sweep_list.txt`.
  Two chats/jobs deploying at once **overwrite each other's source and outputs** (real
  incidents: `sweep_list.txt` cut 48→14 lines; shared `.h5` filenames raced). Before
  dispatching: **check `--status` / `squeue`**; don't launch a second `--option3` sweep
  while another has pending tasks; ensure per-config unique output filenames
  (`generate_file_tag()`), and **serialize** jobs that share mutable state.
  **AMENDED 2026-08-15 (user-approved): sweep lists are now PER-STUDY** —
  deploys write `data/sweep_list_<study>.txt` and export that path, so one
  study's deploy can no longer rewrite the list another study's pending or
  preemption-REQUEUEd task will re-read (the 2026-07-02 killer: hole-scan
  tasks 13–97 died at task-start bounds-check against a 4-task demo's list;
  worse, an in-range index would silently run the WRONG study's row; REQUEUE
  makes even "running-only" queues vulnerable — that is why the old rule was
  absolute). **Parallel deploys are therefore allowed IFF (1) both studies are
  on per-study lists AND (2) the new deploy touches ONLY its own study's
  files** (verify in rsync's itemized output — swapping shared engine/builder
  code under an in-flight study still risks a REQUEUEd task silently re-running
  at different numerics). Any edit to shared code ⇒ serialize as before.
  `--after=<jobid>` chains a dispatch behind an in-flight job (afterok) —
  queue whole stage-sequences in one sitting. Recovery from a clobbered
  legacy-shared list: wait for queue-empty, redeploy, resubmit the dead range
  via `--array-tasks=<lo>-<hi>`.
- **QOS `24h_1g` caps: 100 submitted / 4 running tasks per user.** Arrays >100 tasks
  must go in chunks (`--array-tasks=1-100`, then the rest as the queue drains).
  Count queued tasks with `squeue -r` — plain `squeue` collapses a pending array to
  ONE line and silently undercounts.
- **Stopping runs is a confirm-first action.** Never blanket `scancel`. Resolve the
  specific job ID from `squeue` first, state it back, and confirm before cancelling.
  After cancel, re-check `squeue` to verify. Treat "stop the run" as needing a job ID,
  not speed. (Enforced: `scancel` is on the permission **ask** list — the prompt the
  user approves IS the confirmation. Use the `stop-runs` skill.)
- **Disk quota.** Home has a ~300 GB quota; rebuild-PSO fills it with `.fsp`+`.h5` and
  then jobs silently hang at container init ("Setting --writable-tmpfs"). If jobs hang
  or quota is near 300 G, **delete `.h5` scratch** (don't keep `.h5` by default).
- **Reduce field data server-side before downloading.** The link runs ~0.5–1 MB/s;
  full field-profile `.mat` files are ~650 MB/case while a figure needs one plane at
  one λ (~1 MB). Extract the needed slice on Athena (login-node `python3` has
  numpy/scipy) and download the slice, not the volume (2026-07-02: 2.5 GB pulled for
  4 images before switching).
- **A dispatch request ends with a job ID.** Every "run X" turn ends by stating the
  submitted job/array ID and the task count — or a prominent "NOT dispatched because Y".
  (Real incidents: a requested run silently never submitted, hours lost; a "2-sim"
  comparison quietly dispatched as 5 sims.)
- **★Deploy flags: verify against the parser; code-only push = `--upload-only`.**
  Invented flags were silently ignored TWICE on 2026-08-16 (`--no-submit` →
  stray 10-task array 133070; `--no-dispatch` → duplicate campaign driver
  54440, an hour after the first lesson) while the legitimate `--upload-only`
  existed all along. STRUCTURAL FIX (same day): both deploy scripts now ABORT
  on unknown flags. Residual habit: read the parser before passing a flag you
  haven't used before, and check the queue after every deploy.
- **Silent no-ops.** A license outage makes `fdtd.run()` return instantly with no
  results. If a run finishes implausibly fast / empty, check the license before
  re-dispatching. If a job crashes <30 s right after a config change, suspect **stale
  server code**: restrictive dir perms on remote `project/` can make rsync silently skip
  root `*.py` files (`rsync --inplace` is the known fix — verify the deploy's itemized
  output actually updated the files you edited).
- **Preemption + long drivers (measured 2026-08-14): EVERY Athena GPU partition is
  `PreemptMode=REQUEUE`** — there is no non-preemptible partition. Array sim tasks
  are idempotent (requeue = harmless re-run); any LONG STATEFUL DRIVER (lumopt2
  campaign, optimization loop) must cold-start-resume from its own persisted log.
  Jobs needing >23:30 walltime must submit with `--qos=4d_1g` (or 72h_8g/contrib) —
  the default `ARRAY_QOS=24h_1g` kills them. ★`ARRAY_TIME=...` as an env override is
  **silently IGNORED** (`athena.conf` plain-assigns it after sourcing; `SBATCH_MEM`
  DOES work) — change times via the conf knobs and verify with
  `sacct --format=TimeLimit` after submitting. A third port-expansion-error cause
  (beyond clobber/license below): the sim genuinely never ran or its files landed in
  the container's EPHEMERAL overlay — write sim outputs only under bind mounts.
  Slurm commands work INSIDE the container when needed (recipe + lumslurm configs:
  `memory/project_slurm_container_fixes.md`).
- **★CRITICAL (user rule 2026-08-16, after B4 lost 8.9 h to a REQUEUE): any job
  whose expected runtime exceeds ~2 h MUST persist its progress incrementally
  and resume from it on a cold restart — loss budget on preemption ≤ 1
  evaluation/solve.** Every Athena partition preempts (REQUEUE), so this is
  not optional hardening; an unprotected long job is a DEFECT at dispatch
  time, and losing hours to preemption is a critical incident to be
  root-caused, not shrugged off. The BALANCE (also user): do NOT retreat to
  non-preemptible-only/queue-waiting either — WITH resume, preemptible lanes
  are fine (bounded loss) and short tasks (≤~3 h) may run anywhere,
  preferring the high-priority short-QOS lanes. Resume ≥ lane choice.
- **★LICENSE SEAT CHECK IS MANDATORY before any dispatch of more than one task
  (user rule 2026-08-16), and REACHABILITY ≠ AVAILABILITY.** Ports 1055/2325
  open only proves the server answers; the seat count is what kills runs
  (measured: pool oscillated 39-46/50 within hours). Probe the count from
  IGUM (Athena lmstat is the false negative):
  `$LUM/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve`.
  Budget concurrency vs FREE seats (array task ≈ 1 seat; lumopt2 iteration
  ≈ 2); for long batches keep the trouble-finder seat bands running (≥35/50
  HIGH = hold fan-outs; ≥45/50 CRITICAL = no new dispatches). LocalRunner's
  2 auto-retries are blip-cover, not a plan.
- **License starvation has TWO signatures, one per cluster (measured 2026-08-04).**
  IGUM (native): loud instant death, bare `in run:` + "Unable to checkout". Athena
  (container): SILENT no-op — log shows `Simulation time: ~1 s` and the pipeline later
  crashes with "Can not find result 'expansion for port monitor'". That port-expansion
  error therefore has TWO possible causes: shared-.h5 clobber (see above) OR a license
  no-op — **check the log's "Simulation time" first** to tell them apart (~1 s = license;
  normal solve time = clobber). More rules from the same incident: (a) the 6-concurrent-
  solve ceiling is an UPPER BOUND, not a guarantee — the pool is faculty-shared and both
  our queues being empty proves nothing (4 IGUM + 2 Athena died on seats that "should"
  have existed); (b) N tasks cold-starting an array in the same second can race the
  checkout/ansyscl daemon and the losers die instantly — casualties are cheap, recover
  with a staggered `--array-tasks=<dead indices>` resubmit once the queue drains;
  (c) when opening a SECOND cluster or resuming after any license anomaly, send ONE
  canary task first and confirm a real solve time before committing the fleet.
- **Cluster scripts are a maintained PAIR: athena/ + igum/.** Any edit to
  `athena/scripts/*` or `athena/jobs/*` is either mirrored to `igum/` in the same
  change or explicitly reported as not mirrored. **`dgx/` is FROZEN legacy — do
  not edit it and do not dispatch to it** (broken with 2026R1; see its README
  banner). This rule exists because the forks measurably drifted (2026-07-11
  audit: dgx missing two athena fixes).
- **`lmstat` -96 on Athena is a FALSE NEGATIVE — do NOT block a dispatch on it.**
  `--license-probe` / container `lmutil lmstat` returns `-96` ("lmgrd is not running /
  server down"; locally `HOST_NOT_FOUND`) *even when the license is fully working*. Cause:
  lmstat enumerates by the server's advertised FQDN `lumerical-lm.ece.technion.ac.il`,
  which doesn't resolve — but real jobs check out **by IP** via the `ANSYSLMD_LICENSE_FILE
  =1055@132.68.48.51` / `ANSYSLI_SERVERS=2325@132.68.48.51` env vars the deploy exports.
  So lmstat probes a path real runs never use. Reliable signal instead: TCP ports `1055`
  and `2325` OPEN by IP ⇒ server reachable (open ports + lmstat `-96` = this false
  negative, not an outage); a *genuine* outage no-ops `fdtd.run()` in seconds, so confirm
  with one real sim before concluding "down." (2026-06-30: preflight said "down"; job
  115369 then ran real 7-min solves. Cost a wasted abort cycle.) See
  `memory/project_athena_lmstat_false_negative.md`.

## 7. Don't commit artifacts

Figures and data are regenerated outputs, not source. `.gitignore` covers
`*.mat`/`*.fig`/`*.h5`/`results*/` and now image rasters (`*.png` etc.). Don't `git add`
generated figures or result data; if you see them staged, flag it.

Exception to "regenerated": **convergence-study `.mat` results are keep-forever data**
(expensive to reproduce — a lost TE convergence set forced a full rerun). Never delete
them; when a convergence study finishes, state where the files live.

## 8. Interaction & style

- **An exploratory question is NOT authorization to build or dispatch.** "Can X work?",
  "what should I do?", "מה דעתך" + even a bare "continue" = discuss and propose; do not
  implement new geometry/features or submit jobs until the user picks an option. (Real
  incident 2026-07-01: a "what to do?" question turned into unwanted two-phase-shift
  geometry.)
- **Deleting anything and touching git state require explicit permission.** The
  permission prompts on `rm` / `Remove-Item` / remote `ssh ... rm` / mutating `git`
  commands ARE that request — never route around them (`python -c` with
  `os.remove`/`shutil.rmtree`, output-redirect truncation `> file`, `find -delete`,
  env-prefixed ssh). If a cleanup or git operation is genuinely needed, state exactly
  what would be deleted/changed and let the permission prompt do the asking. This
  includes remote files on Athena (`.h5` scratch cleanup too).
- **Dropped parameters stay dropped.** A parameter/constraint the user removed earlier
  in the session must not reappear in any later plan revision (real incident: tooth
  shift re-added to a TM plan after an explicit "don't do shifts anymore").
- **THE PILLAR PAIR IS PERMANENTLY DROPPED (user rule 2026-08-10, "pillar pair no
  more").** In this project "pillars" means the PERIODIC row of tens of posts (a
  photonic-crystal-like structure) — never the 2-pillar pair. Do not dispatch,
  propose, analyze, or headline the pair in any polarization or study; its stored
  results are historical data only. (Incident: a pair row was included in the TE
  far-field wave 51469 after the user had removed the sparse/pair device.)
- Keep changes minimal and match surrounding code. Don't propose snapshot/auto-save/
  helper-CLI layers on top of workflows that already work via plain file edits.
- Start optimizers from a known-good baseline (regular grating), not multi-start LHS.
- **Links: give the full path, not just a relative one.** When linking to a file, use
  the full absolute path (e.g. `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\matlab_plotting\plot_transmission.m`)
  in the link target, not a bare relative/local path like `matlab_plotting/plot_transmission.m`.
- **End every results/figure answer with the full absolute local paths** to the files
  produced, unprompted (the user has had to ask "give me the full link" 21 times).
- **Plots:** title carries the physical dimensions + resonance λ + peak T; compact
  legends; never label a plot "zoomed"; `'Interpreter','none'` for filename-ish text.
  View naming is deliberately NON-standard: XZ monitor = **"Top view"**, XY monitor =
  **"Side view"** (reverse of the usual convention); x (propagation) always horizontal,
  ux horizontal in far-field plots. Titles short — real π glyph, no mesh/n_core clutter.
  No overlapping tick/exponent labels in stacked subplots. Envelope comparisons =
  envelopes only, overlaid in ONE figure, FWHM in the legend. Final deliverables are
  editable MATLAB `.fig` + PNG (not plotly/matplotlib).
- If the user writes in Hebrew, answer in Hebrew (right-to-left, and avoid em-dashes
  in Hebrew text).

## 9. Honesty & calibration (overrides style, speed, and optimism)

- **Report what happened, not what was hoped.** Failed test, undispatched job,
  skipped step, partial download, empty result — state it first and prominently,
  before any summary of success. "Done" is only for things actually done and checked.
- **Label every quantitative claim** as one of: MEASURED (read from a named file
  this session — cite the file), DERIVED (computed from measured values — show
  from what), or EXPECTED (theory/memory/estimate — say so). Never state numbers
  from a file that wasn't opened this session.
- **No overstatement.** Near-noise-floor effects (§2), single-point results, and
  unconverged optimizations are "candidate"/"preliminary" — never "confirmed",
  "proven", "best", or "significant" until the §2 sanity checks pass. State the
  uncertainty with the claim, not after being asked.
- **"I didn't check" beats a plausible guess.** A confident wrong answer costs
  GPU-hours; "unsure, let me verify" costs a minute. When memory and current code
  disagree, the code wins and the memory gets corrected in the same session.
- **Push back on wrong premises.** If the user's assumption contradicts the data,
  say so directly instead of building on it.

## 10. Code lifecycle — AI-generated study code must not accumulate

(2026-07-11 audit: ~90 spent one-off scripts had piled up in live directories —
23-file side_by_side tree, 8 phase0 gates, 47 job-specific MATLAB plots, 14
near-duplicate runner families — making the repo unusable without a big cleanup.
These rules prevent the re-accumulation, at creation time.)

- **Reuse before creating.** Before writing any new file, check whether an
  existing engine already does it: a sweep is a `SweepSpec` in ONE small file
  (never a copied runner with edits); a plot goes through an existing
  `matlab_plotting/` engine script when one fits. Copy-with-tweak of an existing
  study file is the pattern that created the 14 duplicate families — parameterize
  instead when reasonable.
- **One study = one runner file + at most one plot script**, named after the
  study dir. Every one-off script's header states: study dir, job ID(s), date,
  and one line of purpose. No `_v2`/`_fixed`/second-name copies — edit the
  original (git keeps history).
- **AI scratch/debug code never lands in the repo.** Throwaway test scripts,
  probes, and comparison snippets go in the session scratchpad or get deleted in
  the same session. If it isn't something the user would run again, it doesn't
  get a file in the project.
- **When a study closes, archive in the same session:** its one-off runners →
  `runners/archive/`, its one-off plots → `matlab_plotting/studies/`, unedited
  (they are the lab notebook — never rewrite archived science). Live dirs hold
  only engines + active studies. Verify after moving: deploy-menu listing
  unchanged for live studies + `python -m compileall` clean.
- **Very long new code is a smell, not an achievement.** A new runner over ~150
  lines or a new module over ~400 lines needs a stated reason (e.g. a genuine new
  engine); otherwise decompose or reuse. Never grow the god-objects
  (`bragg_device.__init__`, `deploy_athena.sh`) casually — additions there get a
  one-line heads-up.

## 11. Coding style — write like a lazy senior dev

(Adapted from the "ponytail" skill's decision ladder. §10 says where code lives and
how much may accumulate; this section says what the code itself looks like. The user
is a physicist who will reread and edit this code months later — optimize for that
reader, not for the AI that wrote it.)

- **Climb the ladder, stop at the first rung that holds:** (1) does this code need
  to exist at all? (2) does the codebase already do it? (3) does
  numpy/scipy/stdlib/MATLAB built-ins do it? (4) does an installed package do it?
  (5) can it be a few plain lines? Only then write "the minimum that works".
  Lazy about the *solution*, never about *reading* — understand the existing code
  first; the ladder is not an excuse to skip §5 verification.
- **Compact but not cryptic.** Short means fewer moving parts, not code golf. Prefer
  the boring obvious construct (a plain loop, a plain dict) over a clever one-liner,
  chained comprehension, or lambda pile the user would have to decode. Plain
  functions + module-level CONSTANTS at the top of the file (the knobs a user tweaks)
  beat classes, decorators, and config objects here.
- **No speculative scaffolding.** No CLI flags, config options, plugin hooks,
  abstraction layers, or "for future use" parameters that this study doesn't use
  today. No try/except wrapping that hides errors — in study code a loud stack trace
  is the correct behavior. Add generality the second time it's actually needed, not
  the first time it's imaginable.
- **Structure for the human reader:** one screen ≈ one idea; the file reads top to
  bottom in execution order (config → build → run → save); names say physics
  (`corrugation_nm`, not `param2`); comments only where the *why* isn't in the code
  (units, sign conventions, incident numbers). If a helper is used once and is under
  ~5 lines, inline it.
- **Before reporting "done", reread the diff as the user:** anything you'd have to
  explain in chat should instead be simplified in the code. If the diff is much
  longer than the task sounded, say so and why — length surprises get flagged, not
  buried.
