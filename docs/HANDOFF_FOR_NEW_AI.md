# PROJECT HANDOFF — π-shift Bragg grating FDTD program

**Repo:** `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes`
**Written:** 2026-09-29 · git branch `add-claude-rules-skills` @ `9b8de59`
**Audience:** a new AI assistant (any vendor) or a new human collaborator, with read/write
access to this repo and to the two Technion SLURM clusters, and with **no prior context**.

This file is the complete transfer of what ~40 working sessions accumulated: the device, the
physics results, the code, the cluster operations, and the working rules. It is organised so
you can read Part 0 (20 minutes) and start being useful, then consult Parts 1–4 as reference.

> **Provenance discipline used throughout.** Every quantitative claim in this document is
> labelled implicitly by where it came from: numbers with a job ID or a `.mat`/`.jsonl` path
> are MEASURED; numbers marked "predicted"/"model" are EXPECTED. When you restate any of them
> to the user, keep the label. Never present a remembered number as a freshly measured one.

---

## Contents

**Part 0 — Orientation and working rules** (read this first)
- §0 Orientation: the device, the goal, how work gets done, the cost model
- §1 Correctness rules you must not break (resonance, Q, the two FWHMs, sanity checks, mesh)
- §2 Verification policy: smoke-test, but don't over-test
- §3 Cost discipline and server safety
- §4 Working with this user
- §5 Code and style conventions
- §6 Where the live state lives (the authoritative files)

**Part 1 — The physics: canon, verdicts, devices, models, traps**
- §1 Device and geometry canon
- §2 Closed studies — the verdict table
- §3 The device scoreboard — best known devices
- §4 Physics models and predictive tools (the q3db engine, CMT, Q3dB method, comb model, target locking, radiation rules, width levers, the high-Q adequacy trap)
- §5 Inverse design (lumopt2) — method and state
- §6 Infrastructure facts
- §7 Traps and gotchas — the complete catalogue (285 numbered entries)
- §8 Open threads and next steps

**Part 2 — Cluster operations manual** (hosts, deploy flags, dispatch, status, fetch, job scripts, IGUM differences, failure signatures, adding a study)

**Part 3 — Code and data inventory** (engine modules and the full knob surface, the runners tree, python_tools, MATLAB, result layout, verification gates, repo docs)

**Part 4 — The written recipes** (`.claude/skills/`)

**Part 5 — Current state and a first-hour checklist**

**Companion file — `docs/HANDOFF_APPENDIX_memory_dump.md`** (876 KB, 13,079 lines): the verbatim,
lossless dump of all ~120 memory files this handoff was distilled from. Use it to trace any number
back to its source file and incident narrative.

---

## 0. Orientation

### 0.1 What the device is

A **pi-shift Bragg grating** (always call it that in discussion and writeups — not "phase
shift grating", not "cavity", not "DBR"): a silicon-nitride ridge waveguide whose sidewalls
are corrugated at the Bragg pitch, with a **half-period phase slip at the centre**. The slip
opens a single localised defect resonance inside the photonic stopband. Light launched from
one port tunnels through that defect state and out the other port.

```
   ...[narrow][wide][narrow][wide]  ||π||  [wide][narrow][wide][narrow]...
        N periods, left arm          slip        N periods, right arm
                                      ↑
                     defect mode lives here, ~10-20 µm FWHM along x
```

Key structure facts:
- Material: Si3N4 core, SiO2 cladding. Constant indices are used by default:
  **`n_core = 1.97`, `n_clad = 1.444`** (stable across the whole program).
- Propagation is along **x**; **y** is the lateral (corrugation) direction; **z** is the
  growth direction (core height). Core height default **350 nm**.
- Symmetric device: `n_periods_each_side` per arm ("N=80" always means 80 *per side*).
- The corrugation is described by an **average width** and a **corrugation depth**
  (= wide − narrow), not by two separate widths.

### 0.2 What the program is trying to achieve

The device is meant as an **acousto-optic / acoustic detector element**. That fixes the
figure of merit in an unusual way:

1. **The spatial mode width is a HARD SPEC, not something to minimise.** The acoustic
   interaction needs the optical mode to overlap a specific acoustic footprint, so the
   spec is two-sided: **narrowing the mode does not help**. Typical targets: 20 µm, 14 µm.
2. Subject to that fixed width, **maximise on-resonance transmission T** (equivalently
   minimise radiation loss), and **maximise loaded Q**.
3. A recurring deliverable is the **"Q3dB device"**: the device whose peak transmission sits
   at a chosen dB point (usually **−3 dB, T ≈ 0.50**) at a chosen mode width, with the
   highest Q obtainable there. Q and T trade against each other through device length, so
   "the −3 dB point" is the agreed operating point that makes Q comparable across designs.

Two polarizations are studied, with separate anchored geometries: **TE** (the original) and
**TM** (added later; radiates more, has roughly half TE's light-cone margin).

### 0.3 The three ways work gets done here

| Mode | What it is | Where |
|---|---|---|
| **Parametric sweeps** | one declarative `SweepSpec` per study → a SLURM array, one FDTD sim per combination | `runners/sweeps/` |
| **Predict-then-confirm** | a calibrated semi-analytic engine predicts T/λ/Q/widths for a proposed device; ONE FDTD run confirms against pre-registered bands; the engine is refit | `python_tools/predict_q3db.py` |
| **Adjoint inverse design** | lumopt2 + a 191-parameter design vector, projected-gradient with a hard width constraint | `runners/lumopt2_design/` |

The predict-then-confirm engine is the single biggest productivity gain in the program: it
replaced multi-run tuning ladders with one confirmation run, and it has held on hold-out
backtests (48/50) and on three live predictions. **Use it before dispatching any length or
corrugation tuning ladder.** Coupled-mode theory is *authorized* for that engine; it is
*banned* inside the lumopt2 optimizer (see Part 1 §5).

### 0.4 Cost model — why the rules below are strict

One FDTD solve of the real device is **~1 GPU-hour**. An inverse-design iteration is ~2.5
GPU-h. A long campaign is 10–100 GPU-h. License seats are shared with the whole faculty and
a single GPU solve consumes ~7 of ~42 effective seats, so **at most ~6 concurrent solves
exist in the world** for this project. Every Athena GPU partition preempts by REQUEUE.

The consequence, which is the single most important cultural fact about this project:
**a simulation run is never a trivial action.** The accumulated rules exist because real
incidents cost real GPU-days — a dead TM device ran for 8 GPU-h, a bad gradient burned 30,
a mesh artifact 20, a source-phase bug wasted weeks. Under-testing has repeatedly cost
hours; over-testing has never once cost anything.

---

## 1. Correctness rules you must not break

These are physics/measurement invariants. Violating one silently produces confident garbage.

### 1.1 Resonance and Q

- **Never pick the resonance by `max(T)` or `argmax(T)`.** The global T maximum sits in the
  passband (~1570 nm), not at the defect peak. Use the stored `resonance_wavelength_nm`
  field, or the peak finder in `matlab_plotting/plot_transmission.m` (a sharpness × dip-depth
  scorer inside the stopband).
- **"FWHM" means the SPECTRAL FWHM** (`spectral_fwhm_nm`, from T(λ)) unless the user says
  "spatial". The stored value is **often negative** → always use `|spectral_fwhm_nm|`.
- **Q = `resonance_wavelength_nm` / |`spectral_fwhm_nm`|.**
- **`fwhm_m` is the SPATIAL mode width** (energy vs x) — the acoustic spec quantity. It is a
  completely different number from the spectral FWHM. Confusing the two is a recurring error;
  there is a memory file dedicated to it.
- Mode width is measured **exactly one way**: `sim_helpers.extract_and_process_field_profile`,
  the same convention as `post_processing`'s `fwhm_m`. A raw-line variant, fitted width
  slopes, and a CMT width model were all tried, all wrong, and all deleted by user order —
  **do not reintroduce them**.
- **Every `sigma` / `FWHM` logged by the inverse-design engine before 2026-08-18 is VOID**
  (the extraction never integrated over y). T, λ, Q, R and loss from that era are unaffected.

### 1.2 The mandatory post-run sanity check

Before trusting or building on ANY FDTD result:

1. `resonance_wavelength_nm` exists, is finite, and lies **inside** the scan window.
2. Peak T is above a sane floor. A dead device reads **T ≈ 0.0008**. Healthy TM peaks can be
   as low as ~0.83 in some families, so use a low floor, not a TE-tuned one.
3. If either fails: **stop and say so** ("no resonance found / off-window / dead device").
   Do not build downstream conclusions. A "converged" optimization on a dead device returns
   confident nonsense — this has happened.

### 1.3 Single-wavelength extractions

Monitors and extractions at one wavelength must key off the **index of
`resonance_wavelength_nm`** in the recorded band. Never use "1 frequency point + source
limits" — that records at the band-centre *frequency* (≈1546.4 nm here), not at the
resonance. Far-field was plotted at the wrong λ twice this way.

### 1.4 Comparing absolute numbers

- **Absolute T and loss are numerics-sensitive. Compare only within identical numerics.**
  For strongly-radiating variants, the transverse box size alone moves absolute T by ~3
  points (3.8 → 4.8 µm: 0.828 → 0.799), and the mesher choice moves it again.
- Every sweep carries **its own in-study no-change control at the exact same numerics**, and
  all reported deltas are versus that control. A *stored* identical-numerics control
  satisfies this — do not re-measure it (see §3.2).
- **A candidate effect near the numerical noise floor is not a result.** Measure the floor
  inside the sweep (repeat a few points offset by half a mesh cell), then confirm survivors
  at `simulation_mode="accurate"`. Template incident: a +0.0020 T effect sat exactly at the
  dx = 50 nm jitter floor of 0.0018; at dx ≈ 35 nm the jitter collapsed to 0.0001 and the
  effect survived. That two-step is the standard.
- **Two meshers coexist in this project** (PVA vs conformal). The same device reads λ +5.3 nm
  and FWHM −8% between them. Never compare across meshers. The spec is conformal; PVA is
  used as a first-order gradient tool only.

### 1.5 Mesh policy

- `simulation_mode = "optimization"` (dx = 50 nm) is the **default** and the right choice for
  all sweeps and optimizations.
- `"accurate"` (dx ≈ 35 nm) is reserved for **final / fab-comparison validation**. It is
  case-dependent, never automatic.

### 1.6 Geometry defaults are DEFAULTS, not constants

Indices are stable. The anchored TM geometry is **per core height** and has changed several
times: for height 350 nm it is pitch **516.83 nm**, corrugation **400 nm** (co-resonant with
TE and width-matched). **Pitch and corrugation are coupled — change one and re-trim the
other.** At the start of any new TM task, *confirm height + pitch + corrugation with the
user* rather than assuming. Baselines: TE N = 80/side; TM period-matched to TE@80 is
N = 132/side.

When a material index or pitch changes mid-study, **re-scan the baseline** at the new
resonance. Reusing the old scan window is how peaks get missed. Before dispatching a new
scan, state the target λ and the window width in one line and sanity-check them (past
incidents: a 75 nm window where ~20 nm was meant; aiming at 1449 nm when 1550 was meant).
Pitch-retune acceptance default: present the residual detuning Δλ and accept when it is
≲ 1 nm.

---

## 2. Verification policy — smoke-test, but don't over-test

**Smoke-test before dispatch when the change touches:** (1) device geometry, (2) a new
builder or parametric scaffold, (3) inverse-design / gradient equations, (4) source or
boundary-condition setup. Especially anything new.

Concretely:

- **Lumapi:** local build-only `save_fsp` (<1 min) and eyeball the geometry. All local
  verification is **silent** — lumapi always `hide=True`, MATLAB always `-batch`. Nothing
  opens a window on the user's screen during an automatic step.
- **The config-override trap:** `SimulationConfig` dataclasses accept UNKNOWN attributes
  silently. `cfg.grating.corrugation_depth_m = ...` creates a dead attribute (corrugation
  lives on `cfg.geometry.*`) and the device builds at the default. After any direct-attribute
  override, verify the built values via `SPEC.expand()` / `describe()` / a build printout.
- **Any edit to `bragg_device.py` geometry or monitor code:** run
  `PYTHONIOENCODING=utf-8 python debug_fsp_compare/scene_snapshot.py --out <tmp>` and diff
  against the committed references in `debug_fsp_compare/snapshots/` (6 configs spanning the
  builder's code paths; byte-identical = behaviour preserved). Regenerate the references only
  when a geometry change is intended, and say so.
- **New gradient method:** finite-difference `check_gradient` on a tiny problem first; the
  hard gate is a small `vec_error`.
- **A math gate is not a plumbing gate.** Drive new FOM/jacobian/adjoint code through the
  REAL wrapper locally (build the actual fct, call `autograd.jacobian` on it) before
  dispatch. A gate that *cannot fail* proves nothing — assert that the known-bad form still
  raises. Incident: a job died at 2:03 on `IndexError` while its math gate passed at 0.0034%,
  because the fct's `x` is the FLAT vector `[T(λ_0)…T(λ_n), softW]`, not a list of FOM
  entries.
- **Verify a new feature's ENGAGEMENT CONDITIONS on paper before the test run.** List every
  trigger (count thresholds like "refit at n ≥ 5", eligibility windows, state flags) and
  check the validation run actually reaches them. Two burns in one day came from this: a
  crash inside a refit that engages at n ≥ 5 survived every gate because all gates ran n ≤ 4;
  and a smoke test whose eligibility gate could never open at its own operating point. A
  smoke must assert its feature's own log marker fired.
- **Designed recovery paths get an end-to-end smoke through the real wrapper stack.** lumopt2
  double-wraps exceptions (one site without `from e`, one with), so a guard tested at its
  raise site never matched in production — walk both `__cause__` and `__context__`.
- **Validate the parameter vector against its own bounds before every dispatch.** lumopt2
  rejects an out-of-bounds seed outright and the job dies in ~60 s after queueing behind
  everything. Reusable checker: `runners/lumopt2_design/gates/predispatch_check.py`.
- **MATLAB:** `checkcode` lint + a headless `exportgraphics` render.

**Two scaling rules that save whole evenings:**

- **Debug on the smallest scene that can answer the question — never on the device.** A
  question about numerics, an API, a solver limit, a crash signature, or a launch/config
  error does **not** depend on our grating: it needs an empty box, a dummy source, a short
  sim time, and it answers in seconds. Only device-physics questions (T, λ, Q, mode width,
  gradients of those) need the real device, and even then prefer the smallest N that keeps
  the physics. Incident: a CUDA kernel-launch bound was chased with full-device rungs at
  45–70 min each across four jobs (~6 GPU-h) for a question with nothing to do with the
  grating. Bisect a threshold in ONE array of cheap tasks, never one expensive point per
  dispatch.
- **Hardware-touching engine changes get a minutes-scale end-to-end pass before any
  hours-scale dispatch.** Local gates catch math and call-path bugs but not
  live-session-state bugs. Order jobs so new code executes earliest (fail fast beats fail
  late); prefer many short discriminating runs over one long confirmatory one; when designing
  any validation, first ask "what is the cheapest run that can kill this?".

**Skip** re-verifying known-good baselines and re-linting untouched code. Don't invent extra
test passes for mechanical edits.

---

## 3. Cost discipline and server safety

### 3.1 Think first, run second

Before ANY dispatch: state **what the run will decide** and **why existing results can't
answer it**, and prefer the smallest discriminating experiment. After ANY anomaly (unexpected
λ/T/fwhm, a <30 s crash, an off-family value): **no new runs** until the cause is understood
from free diagnostics — stored `.mat` comparisons, scene diffs, job/solver logs, local
build-only rebuilds.

**A dispatch request ends with a job ID.** Every "run X" turn ends by stating the submitted
job/array ID and the task count — or a prominent "NOT dispatched because Y". Real incidents:
a requested run silently never submitted (hours lost); a "2-sim" comparison quietly dispatched
as 5 sims.

### 3.2 Never re-measure a stored result

This is the rule the user has enforced most often ("if we have a result somewhere don't do
it again — very important").

- Before any dispatch, enumerate which requested points already exist (`results_from_athena/`,
  `results_from_igum/`, the memory files, the eval logs) and **cut them**. The dispatch note
  says "point X reused from `<job/file>`".
- Default = **no control row**; cite the stored baseline file instead.
- **A stored result's identity = engine version + numerics (mesh/dx, mesher, window/points,
  box, boundary conditions) + spec parameters. The cluster/machine is NOT part of the
  identity** — cross-cluster reproducibility is proven exactly (corr-325 N165 control
  T 0.4906 / Q 13930 identical on both clusters). A cluster switch alone never justifies a
  re-run. Only a *named* numerics change or an engine bump does.
- **"I can't verify it's identical" is NEVER a reason to re-run.** Verification is cheap local
  work — the stored `.jsonl`, the runner docstring, the job log, and the handoff docs carry
  the version and numerics. Go read them. Re-run only when a real difference is found, or
  provenance is genuinely unrecoverable AND the number is decision-critical.
- **Optimizer lanes obey the same rule.** A campaign continuing a prior lane must *inherit*
  its state (copy `<label>_evals.jsonl` + `<label>_optstate.json` into the new label's
  out_dir server-side before dispatch) so it warm-starts instead of re-deriving iterates at
  ~2.5 GPU-h each. Never dispatch a separate seed/benchmark re-measure.
- Corollary duty: every result you store or cite carries its engine version and numerics, so
  this check stays a 2-minute read.

### 3.3 Preemption and long jobs

**Every Athena GPU partition is `PreemptMode=REQUEUE`** — there is no non-preemptible lane.
Array sim tasks are idempotent (a requeue is a harmless re-run). But **any job expected to
run longer than ~2 h MUST persist progress incrementally and resume from it on a cold
restart**, with a loss budget of ≤ 1 evaluation. An unprotected long job is a **defect at
dispatch time**, and losing hours to preemption is a critical incident to root-cause, not
shrug off. The balance: *with* resume, preemptible lanes are perfectly fine — resume beats
lane choice, and you should not retreat into queue-waiting for a "safe" partition.

### 3.4 License seats

**A seat check is mandatory before any dispatch of more than one task, and reachability ≠
availability.** Ports being open only proves the server answers; the seat count is what kills
runs (the pool has oscillated 39–46 of 50 within hours). Probe the count *from IGUM* (Athena's
`lmstat` gives a documented false negative). Bands: ≥35/50 in use = HIGH, hold fan-outs;
≥45/50 = CRITICAL, no new dispatches. Budget ~1 seat per array task, ~2 per inverse-design
iteration — but note the measured reality that a GPU solve takes ~7 seats, giving ~6
concurrent solves total across both clusters.

**License starvation has two different signatures, one per cluster** (see Part 2 §7):
IGUM dies loudly and instantly; Athena **silently no-ops** with `Simulation time: ~1 s` and
crashes downstream with "Can not find result 'expansion for port monitor'". That same
downstream error also means a shared-`.h5` clobber — check the log's "Simulation time" first
to tell them apart.

### 3.5 Concurrency, disk, and data

- Deploy does `rsync --delete` into a **shared** remote `project/` and writes into a shared
  `results/`. Two chats deploying at once overwrite each other's source and outputs (this has
  happened). Sweep lists are now **per-study** (`data/sweep_list_<study>.txt`), which makes
  parallel deploys safe **iff** both studies are on per-study lists and the new deploy touches
  only its own study's files (verify in rsync's itemized output). Any edit to shared engine
  code ⇒ serialize.
- QOS `24h_1g` caps **100 submitted / 4 running** tasks. Count queued tasks with
  `squeue -r` — plain `squeue` collapses a pending array to one line and undercounts.
- Home has a **~300 GB quota**. Exceeding it makes jobs silently hang at container init
  ("Setting --writable-tmpfs"). Delete `.h5` scratch; don't keep `.h5` by default.
- **Reduce field data server-side before downloading.** The link runs ~0.5–1 MB/s; a full
  field-profile `.mat` is ~650 MB while a figure needs one plane at one λ (~1 MB). The login
  node's `python3` has numpy/scipy — slice there and download the slice.
- **Never let one cluster hold unique results.** A campaign's incremental log is unique data
  the moment it is written. Pull the small state files (`*.jsonl`, `*.csv`, ~KB) on every
  milestone check, not at study end. (The "reduce server-side" rule concerns big field
  volumes, never these.)
- **Login-node connection budget: ≤ ~3–6 ssh/hour per cluster for automated polling, ONE
  connection per poll** (fold queue + log + license probes into the same ssh). IGUM began
  refusing the key ~80 min after a monitor polled it 24×/h; ~45 min of zero contact restored
  it. On any auth refusal with port 22 open: stop all automated contact for ≥45 min, then ONE
  probe — never a retry loop. Cluster *jobs* are unaffected by login-node auth, so an outage
  costs visibility, not science; never panic-redispatch because a login node is refusing.
- **ssh command form is always host-first:** `ssh evyatarrubin@athena.technion.ac.il "..."`.
  Never an env-var-prefixed form (`SSHHOST=... ssh "$SSHHOST" ...`) — those evade the
  permission-rule pattern matching, including the `scancel` guard. Strip the Technion banner
  with `grep -vE "post-quantum|openssh|may need to be upgraded"`.
- **Stopping runs is confirm-first.** Never blanket `scancel`. Resolve the specific job ID
  from `squeue`, state it back, confirm, cancel, then re-check `squeue`.

---

## 4. Working with this user

The user is a **physicist** (Technion, Hebrew/English bilingual) who will reread and edit this
code months from now. Optimise every artefact for that reader.

**Communication**
- **Short answers.** Lead with the verdict or state in 1–3 sentences. Status updates are 1–2
  lines. Bullets only for things the user must act on. Details live in the handoff/skill
  files — link them rather than restating.
- Campaign status lines always carry the physics deltas (per-iterate ΔW in µm and ΔT, step and
  cumulative versus seed) and quote **t_pk (peak transmission), λ_pk and W per eval** — the
  optimizer's `fom` alone is never enough.
- **If the user writes in Hebrew, answer in Hebrew** (right-to-left; avoid em-dashes in
  Hebrew text).
- **Label every quantitative claim** MEASURED (read from a named file this session — cite it),
  DERIVED (computed from measured values — show from what), or EXPECTED (theory/estimate).
  Never state a number from a file that was not opened.
- **Report what happened, not what was hoped.** Failed test, undispatched job, skipped step,
  partial download, empty result — state it first and prominently. "Done" is only for things
  actually done and checked. Near-noise, single-point and unconverged results are
  "candidate"/"preliminary", never "confirmed", "proven", "best" or "significant".
- **Push back on wrong premises.** If an assumption contradicts the data, say so directly
  instead of building on it. "I didn't check" beats a plausible guess. When memory and the
  code disagree, **the code wins** and the memory gets corrected in the same session.
- **Don't call the best device "champion."** Name devices by geometry ("rect-1050", "the
  1050+see-saw stack", "corr-325 N172 + r110 comb pair").

**Decision boundaries** (these two rules look contradictory; they are not)
- **An exploratory question is NOT authorization to build or dispatch.** "Can X work?",
  "what should I do?", "מה דעתך", and even a bare "continue" mean *discuss and propose*.
  Do not implement new geometry/features or submit jobs until the user picks. A real incident:
  a "what to do?" question became an unwanted two-phase-shift geometry that had to be ripped
  out. Likewise, a **pace complaint is not authorization for a strategy pivot** — deliver the
  re-think as a recommendation and dispatch only after explicit approval.
- **But inside an approved plan, do not present menus — decide, execute, report.** In the
  inverse-design programme especially, the user reads an options list as "the model has lost
  the plan". Pick the path using the measured evidence, state the decision in one line with
  its reason, then do it. A genuine fork gets a *recommendation* plus what would change it.
  Fixing broken things inside the approved plan (resubmits, gates, cleaners) is autonomous;
  changing *which question the GPUs are answering* is not.

**Scope hygiene**
- **Dropped parameters stay dropped.** A parameter or constraint the user removed earlier must
  not reappear in a later plan revision (a tooth shift was re-added to a TM plan after an
  explicit "don't do shifts anymore").
- **The pillar PAIR is permanently dropped.** In this project "pillars" means the *periodic
  row* of tens of posts; the 2-pillar pair is dead everywhere — do not dispatch, propose,
  analyse or headline it. Its stored results are historical data only.
- **A yes/no physics question gets the minimal sim count**: one configuration at the known-best
  parameters plus its identical-numerics control — not a parameter bracket "while we're at it".
  Brackets and refinements are a *second* dispatch after the yes/no lands.
- **No pre-baked contingency sections in plans.** Commit to one approach; if it fails, raise
  it then.
- **No speculative scaffolding.** No snapshot/auto-save/helper-CLI layers on workflows that
  already work through plain file edits — the file *is* the persistence.
- **Uncertain structural counts must be optimizable.** When the user says a discrete quantity
  is unsettled ("the number of posts could be lower or higher"), that uncertainty gets an
  exploration mechanism (a free parameter, a continuous relaxation, or a scheduled count
  ladder) — it does not get frozen at the last measured optimum.
- **Deleting anything and touching git state require explicit permission**, including remote
  files. Do not route around a refusal with `os.remove`, `shutil.rmtree`, `> file`
  truncation, `find -delete`, or an env-prefixed ssh.
- **Don't commit artifacts.** Figures and result data are regenerated outputs, not source.
  Exception: **convergence-study `.mat` results are keep-forever data** (a lost TE convergence
  set forced a full rerun).

**Lesson capture is a standing duty.** Every incident, measured limit, surprise or correction
gets written into the rules/handoff/memory **in the same session, unprompted**. The test:
would the next session repeat the mistake? Adversarial forethought is a design-time duty too —
before dispatching new code, silently run the how-can-this-fail pass over engagement
conditions, environment differences (numpy / container-vs-native / API versions on the *target*
machine) and state boundaries (restart, REQUEUE, resume, label inheritance). A fix that is
blocked on data or capacity is **owed, not closed** — carry it with its blocker and its
clearing trigger, and land it the moment the trigger fires.

---

## 5. Code and style conventions

**Code lifecycle** (an audit once found ~90 spent one-off scripts piled up in live
directories, making the repo unusable without a big cleanup):
- **Reuse before creating.** A sweep is a `SweepSpec` in ONE small file — never a copied
  runner with edits. A plot goes through an existing `matlab_plotting/` engine when one fits.
  Copy-with-tweak is what created 14 near-duplicate runner families; parameterize instead.
- **One study = one runner file + at most one plot script**, named after the study dir. Every
  one-off script's header states: study dir, job ID(s), date, one line of purpose. No
  `_v2` / `_fixed` copies — edit the original; git keeps history.
- **Scratch/debug code never lands in the repo.** If the user would not run it again, it does
  not get a file.
- **When a study closes, archive in the same session**: one-off runners → `runners/archive/`,
  one-off plots → `matlab_plotting/studies/`, **unedited** (they are the lab notebook — never
  rewrite archived science). Verify afterwards: the deploy-menu listing is unchanged for live
  studies, and `python -m compileall` is clean.
- **Very long new code is a smell, not an achievement.** A new runner over ~150 lines or a new
  module over ~400 needs a stated reason. Never grow the god-objects
  (`bragg_device.__init__`, `deploy_athena.sh`) casually.

**Coding style — "write like a lazy senior dev"**: climb the ladder and stop at the first rung
that holds — (1) does this code need to exist? (2) does the codebase already do it? (3) does
numpy/scipy/stdlib/a MATLAB built-in do it? (4) does an installed package? (5) can it be a few
plain lines? Be lazy about the *solution*, never about *reading* the existing code. Compact but
not cryptic: prefer a plain loop and a plain dict over a clever one-liner. Plain functions plus
module-level CONSTANTS at the top of the file (the knobs a user tweaks) beat classes, decorators
and config objects. No try/except that hides errors — in study code a loud stack trace is
correct. Structure for the human: one screen ≈ one idea, the file reads top-to-bottom in
execution order (config → build → run → save), names say physics (`corrugation_nm`, not
`param2`), comments only where the *why* isn't in the code (units, sign conventions, incident
numbers).

**Plot conventions** (the user has strong, deliberate preferences):
- Title carries the physical dimensions + resonance λ + peak T. Compact legends. Never label
  a plot "zoomed". `'Interpreter','none'` for filename-ish text.
- **View naming is deliberately NON-standard: the XZ monitor is the "Top view" and the XY
  monitor is the "Side view"** — the reverse of the usual convention. Follow it.
- x (propagation) is always horizontal; ux is horizontal in far-field plots.
- Titles short, real π glyph, no mesh/`n_core` clutter. No overlapping tick/exponent labels in
  stacked subplots. Envelope comparisons = envelopes only, overlaid in ONE figure, with FWHM in
  the legend.
- Final deliverables are **editable MATLAB `.fig` + PNG** — not plotly, not matplotlib.

**Links and paths:** always give the **full absolute path** when linking a file, and **end
every results/figure answer with the full absolute local paths** to the files produced,
unprompted. (The user has had to ask "give me the full link" 21 times.)

---

## 6. Where the live state lives

Read these before touching the corresponding work. They are the authoritative, editable state;
this handoff is a snapshot.

| File | What it holds |
|---|---|
| `CLAUDE.md` | The always-on invariant rules, in their canonical form (Part 0 above is the distillation of it) |
| `README.md` | Architecture, config groups, the apodization/shift/experiment-card interfaces, output `.mat` field list |
| `runners/README.md` | The two study patterns and the **deploy-menu discovery contract** (what makes a file appear in the cluster menus) |
| `runners/lumopt2_design/HANDOFF.md` | Live state of the inverse-design programme (3372 lines) |
| `runners/lumopt2_design/HANDOFF_2026-09-01.md` | The **current** start-here handoff for that programme (241 lines) |
| `runners/lumopt2_design/HANDOFF_SELF_CONTAINED.md` | Method + the full 191-parameter design vector + code + raw data — hand this to a session with no repo access |
| `runners/lumopt2_design/THEORY.md` | The METHOD (not the state): cost function, parameter layout, the two width measures, projected-gradient algorithm, how the adjoint gradients are obtained, the resonance chain-rule term |
| `runners/lumopt2_design/DESIGNS.md` | The stored design vectors |
| `runners/scatterers/COMB_HANDOFF.md` | The comb / anti-needle scatterer programme |
| `docs/calibration_dossier_2026-09-26.md` | The q3db predictive engine's calibration dossier (975 lines) |
| `docs/comb_physics_rethink_2026-09-11.md` | The comb physics rethink that produced the newest Q3dB device |
| `docs/research_overview_briefing_2026-09-13.md` | Advisor-level overview of the whole program |
| `FILE_NAMING.md` | The result-filename convention |
| `docs/HANDOFF_APPENDIX_memory_dump.md` | The lossless dump of the ~120-file memory store this handoff distils (876 KB) — the provenance layer behind every number in Part 1 |

**Read `THEORY.md` before reasoning about the optimizer or its gradients; read the newest
`HANDOFF_*.md` before running anything in that programme.**

---

# Part 1 — The physics: canon, verdicts, devices, models, traps

Everything in this part is distilled from the project's persistent memory store (~120 files,
~1 MB, listed in Part 4). Numbers carry their job IDs and source files so any of them can be
traced back. Sections are numbered §1–§8 within this part.


## 1. Device and geometry canon

### 1.1 What the device is called, and what it is

- **Always call it a "pi-shift Bragg grating"** — a Bragg grating with a π/2 cavity-length defect
  → π round-trip phase shift → transmission notch (defect peak) inside the stopband. Repo name
  `phase_shift_grating_FTDT_codes` and class `PiShiftBraggFDTD` are file-level identifiers only;
  do not rename code symbols, do not say "phase-shift grating" in discussion.
  (project_device_terminology)
- Application: **acousto-optic / acoustic detector**. The ~20 µm spatial mode is the acousto-optic
  interaction region. FoM = max on-resonance transmission at FIXED spatial mode width, usually
  quoted at the −3 dB operating point. (project_acoustic_detector_width_spec)

### 1.2 STABLE CONSTANTS (do not vary without being told)

| quantity | value | notes |
|---|---|---|
| `n_core` (SiN) | **1.97** | project-wide default since 2026-06-28 (user: "save for all projects") |
| `n_clad` (SiO2) | **1.444** | stable; `1.44` was a legacy typo cleaned out of live code |
| `shift_bounds_nm` | **(0, 200)** | leaves 50 nm min narrow tooth (250 − 200). **Do NOT tighten.** Validity constraint is `shift_d < half_pitch − fab_min ≈ 220 nm` |
| core height (added structures) | **350 nm** | single litho layer; taller variants are diagnostics only, never device candidates |
| `simulation_mode` default | `"optimization"` (dx = 50 nm) | `"accurate"` ≈ dx 35 nm reserved for final/fab validation |

Index history that still appears in files (all three regimes coexist — confirm before any run):
**1.977** = older project-wide default (side-by-side runners keep it EXPLICITLY);
**1.9963 / 1.444** = the legacy IT11-calibrated TM pin in `runners/tm/_tm_vs_te_common.build_base_cfg`,
`tm_mode_loss.py`, `calibrate_neff.py`, `PITCH_ALIGNMENT.md`; **1.97 / 1.444** = current default.
Deliberately NOT migrated: `compare_8_devices.py`, `compare_tm_optimized_shift`,
`validate_te_scaling.py` (its whole point is the 1.977↔1.9963 scaling ratio).
(project_tm_material_indices, project_tm_pitch_redo_1p97)

### 1.3 DEFAULTS that get changed — TE

| quantity | value | status |
|---|---|---|
| pitch | **500 nm** (half_pitch 250 nm) | TE baseline default |
| corrugation depth ("DW") | **300 nm** for the regular grating | per-tooth parameter |
| average tooth width / `cavity_width` | **800 nm** (y-extent of the waveguide) | default |
| N periods each side | **80/side** ("N periods" ALWAYS means `n_periods_each_side`; "80 periods" = 80 per side) | TE baseline |
| apodized empirical-good start | `[dw_inner_1, dw_inner_2, shift_1, shift_2, cavity_width] = [250, 280, 50, 30, 800]` nm | |

TE baseline observables (era-dependent — never mix):
- n 1.977 / pitch 500 / corr 300 / N80: λ = 1570.7 nm, T = 0.86, `fwhm_m` = 15.2 µm.
- n 1.9963-era matched comparison: TE@80 peak T = 0.8304, spectral FWHM 0.917 nm, Q ≈ 1712, λ = 1571.00 nm.
- n 1.97 / pitch 500: **λ_TE = 1558.74 nm, T 0.87**; TE@corr300 mode width **15.54 µm** (the κ-match target);
  a second reading of the same family gives 15.544 µm / T 0.870 / spectral 1.17 nm.
- TE −3 dB / 20 µm operating device: **N = 166/side, corr 250 nm, T 0.4919 (−3.08 dB), Q_L 12,903,
  fwhm 20.46 µm, λ 1559.79 nm**.
(project_grating_geometry_facts, project_apodization_sweep_tm_te, project_tm_corrugation_match_modewidth,
project_tm_period_match_te, project_te_q3db_20um)

### 1.4 DEFAULTS that get changed — TM, PER CORE HEIGHT

**TM anchored geometry is per-height and is a DEFAULT, not a constant. At the start of any new TM
task, confirm height + pitch + corrugation rather than assuming.**

**h = 350 nm (the standard device):**

| quantity | value |
|---|---|
| pitch | **516.83 nm** (anchored, co-resonant with TE at corr 400) |
| corrugation | **400 nm** (κ-matched to TE's mode width) → tooth widths 600 / 1000 nm |
| avg width / cavity | **800 nm** (W800) |
| N period-matched to TE@80 | **132/side** |
| standard short/surrogate N for corr-400 | **80/side** (2κL = 3.64) |
| corr-325 family | pitch 516.83, production **N = 165–170/side**, surrogate **N = 100** (2κL = 3.65) |
| κ (MEASURED) | corr-325 **0.0353 µm⁻¹**, corr-400 **0.0440 µm⁻¹** (ratio 1.246 vs corr ratio 1.231) |
| mode-width asymptote F∞ | corr-325 **19.87 µm**, corr-400 **16.13 µm** (ln2/κ = 15.75 under-estimates by ~2 %) |

TM pitch lineage (each supersedes the last):
**518.3 nm** at n 1.9963 → **516.14 nm** at n 1.97 with corr 300 (λ_TM = 1558.740 nm, exactly on the
TE target; from a dense 6-point FDTD regression, slope 2.483 nm-λ/nm-pitch, RMS 5.7 pm; confirm run
job 113301) → **516.83 nm** at corr 400 (job 113814: λ 1558.46 vs TE 1558.34, Δ 0.12 nm, fwhm 15.57 µm).

**★PITCH ↔ CORRUGATION ARE COUPLED — change one, re-trim the other.** Raising corr 300→400 detuned
TM down ~1.7 nm (measured slope ≈ **−0.013 nm-λ per nm-corr** at fixed pitch); recovered with
**+0.69 nm of pitch** (λ-sensitivity **2.48 nm/nm**). The two calibrations are NOT orthogonal.
Pitch-retune acceptance default: present the residual Δλ, accept at ≲1 nm (user accepted 0.75 nm).
(project_tm_corrugation_match_modewidth, project_tm_pitch_redo_1p97, project_tm_nladder_surrogate)

TM baseline observables (h350):
- pitch 500 / n 1.97: λ_TM = 1516.34, T 0.96 (Δ 42.4 nm from TE).
- pitch 518.3 / N80: λ 1570.5, T 0.9584, `fwhm_m` 19.03 µm. Pitch-500/N80: λ 1523.6, T 0.97, fwhm 17.9 µm.
- period-matched **N = 132/side** (crossing interpolated 131.6): peak T 0.8285 (ΔT −0.0019 vs TE@80),
  spectral FWHM 0.320 nm, **Q ≈ 4910**, λ 1570.75 nm → **~2.9× TE's Q at the same peak T**. TM@80 Q ≈ 803.
- corr-400 N80 W800 control (converged box y6.8 / z8.8, opt mesh): **T 0.8862, λ 1558.611–1558.617,
  Q_L ≈ 1311–1320, mode 15.53 µm**, resonant loss ≈ 0.110–0.114.
- corr-400 N80 at the default 3.8 µm box (unconverged): T 0.799 / loss 0.19 — a z-PML artifact; the
  converged truth is T 0.886 / loss 0.110.
- corr-325 N=165 production control: **T 0.4906, Q 13,930, mode ~19.97–20 µm, λ 1559.00**.
- corr-325 N-ladder (bare, q3db numerics): N 60/70/80/100/120 → T 0.9674/0.9624/0.9524/0.9104/0.8441,
  Q 395/579/845/1760/3554, mode 16.80/17.74/18.39/19.24/19.66 µm, λ 1559.006–1559.011.
- corr-400 N-ladder: N 60 → T 0.9379 / Q 535 / 14.57 µm; N 70 → 0.9186 / 844 / 15.15 µm; N 80 (stored)
  → ~0.886 / ~1320 / 15.5 µm; λ 1558.616 at both measured rungs.
- λ is **N-independent** (5 pm over N 60→120) — λ is a **pitch-only** knob. Q ∝ exp(2κL).
- Surrogate rule: **match 2κL, not N. Target 2κL ≳ 3.5, hard floor 3.2** (N ≈ 3.5/(2κΛ)).

**h = 200 nm, w = 1800 nm (a different device class):** corrugation 400 nm, pitch **504 nm**,
N = 1300/side, `width_port = 1800` (matches grating avg, no port step). FINAL: λ_res **1492.124 nm**,
**Q = 7.0×10⁵** (τ_E 549–567 ps, radiation-limited), **resonance is DARK, peak T = 0.0031** measured
(DERIVED true T ~0.02), R at peak 0.89, stopband 1490.3–1494.0 nm (~3.7 nm), κ ≈ 10.7/mm, spatial
mode FWHM **547.6 µm**. dz is hardcoded `core_height/7` → a ~50 nm FDTD-vs-FDE λ offset at h200.
**h200 wide-mode variants:** 1550 nm target → pitch **531.5 nm**, corr **227.7 nm** → 80.18 µm mode
(Q ~1659, T 0.71, ~16 % radiation); 1590 nm target → pitch **545.959 nm**, corr **262.7 nm** →
**79.75 µm** mode, λ 1590.02, Q 1554, T 0.562, R 0.032, loss 0.406. Longer λ needs DEEPER corr for the
same width (κ weaker at longer λ) and radiates 41 %.
(project_tm_h200_w1800_study, project_tm_wide_mode_corr)

### 1.5 Cavity detuning and cavity-width options

- `cavity_neg_detuning_nm` — Athena's standing default override is **5.76 nm**. For IT11
  as-fabricated sims it is **0.0** (simulate the geometry verbatim; do NOT inherit 5.76).
- `cavity_width_option` (3-way enum, `bragg_device.py:481-493`), IT11 logic:
  apod off (any ts) → `"narrow"`; apod on and ts == 0 → `"avg"`; apod on and ts > 0 → `"avg_ext"`
  (widens only the d=1 innermost narrow segment after the cavity).
- `shift_target` = `"narrow"` (legacy default) | `"wide"`. A tooth shift `s` bundles THREE changes:
  duty cycle, local period (2HP − s), and **cavity length (+2Σs, common-mode)**. A *negative* shift is
  NOT "shortening the wide part" — it moves the period the other way (2HP + s) AND shrinks the cavity.
(project_it11_device_naming, project_shift_target_sign_test)

### 1.6 The width spec — why it is two-sided

- **The user works at a FIXED width** (2026-08-12): *"narrowing the mode doesn't really help me; it's
  more about keeping it constant and obtaining a high Q / lower radiation."*
- Penalty form: `F = softmax_p(T) − β·max(0, |σ/σ_ctrl,N − 1| − 0.02)²`, **β ≈ 15–20** (a 5 % width
  violation ≈ the whole expected T gain ~0.015). This SUPERSEDES an earlier one-sided form.
- **Narrowing does not help** — it is not the acousto-optic figure; the spec is constancy.
- **Widening is FORBIDDEN** even though it is the biggest Q lever physics offers:
  EXPECTED **Q_i ∝ L_mode²** (k-space: narrower mode → broader k-spread → more weight in the light
  cone; real-space: grating radiates ∝ κ² per length and L_mode ∝ 1/κ). So the h200/w1800 Q ≈ 7×10⁵
  wide-mode regime and every "just delocalize the mode" proposal are OFF the table regardless of Q.
  **Do NOT re-propose widening.**
- Consequence: the ONLY remaining axis is **less radiation at fixed width** → that is why the
  decoration family (comb / trench / per-tooth shaping) is the whole inverse-design scope.
- Width must be compared as a **RATIO σ/σ_ctrl at the SAME surrogate N**, never as an absolute 20 µm:
  the natural bare mode at N=100 (corr-325) is 19.24 µm, so an absolute-20 target would force ~4 %
  artificial κ-weakening. Second-moment truncation beyond the device end (DERIVED, 2κ = 0.0706 µm⁻¹):
  N=80 44 %, N=90 36 %, N=100 29 %, N=110 24 %, N=165 6 %.
- Truncation of the width itself (fraction of asymptote), corr-325: **84/89/92/96/98 %** at
  N = 60/70/80/100/120. corr-400: 94 % (N60) / 98 % (N70) / 96.1 % (N80, NOT 100 %).
- Mode width is **box-INDEPENDENT** (19.24–19.25 µm across 7 boxes; 17.61 µm at every Itai box).
- N frozen + width pinned ⇒ κ pinned ⇒ Q_c ~frozen ⇒ peak T is a monotone reader of Q_i. The acoustic
  spec and the well-posedness of the cost function are the same constraint.
(project_acoustic_detector_width_spec, project_tm_nladder_surrogate)

### 1.7 Naming / file conventions

- **IT11 fabricated-device subname tokens** (`device_names.csv`, parsed by
  `runners/experiment_comparison/it11_card_builder.py:parse_subname`):
  `corrN` → `geometry.corrugation_depth_m = N·1e-9`; `NpN` → `grating.n_periods_each_side`;
  `NtN` → `apodization.{enabled=True, n_apod_periods_each_side=N}` (absent ⇒ disabled);
  `dwmN` → `apodization.center_mod_depth_nm`; `tsN` → `grating.innermost_tooth_shift_m` (absent ⇒ 0);
  `tanh_aX` + `dpm-1e5` → Itay's chirped designs, **skip** (user instruction).
  Each IT11 device → TWO sims, pitch 500 and pitch 516 (the two Bragg resonances ~1535 and ~1577 nm).
- Result-file tags come from `sim_helpers.generate_file_tag`: `_TM`, `_C{corr}`, `_A{n}` apod,
  `_M{dwm}`, `_S{shift}` (+`w` for wide-target), `_dsh…` for list shifts, `_fc` (lengthen_cavity=False,
  scalar path only), `_scR{r}_X{x}_Y{y}_pair` / `_hole` (scatterers), `_scRECT_…_H{nm}` (trenches),
  `_Zminm3975`, `_Ybox…_Zbox…`, `_AS{..}` (auto-shutoff), `_ff`, `_smp`, `_avg2W{nm}`,
  two-device `_2pishift_p{pitch}_Ygap{..}nm_Xstag{..}nm_corr1{..}nm_corr2{..}nm[_closed]`.
  **NOT in the tag: mesh mode, window centre** — same-geometry rows at two meshes COLLIDE on filenames.
- Plot view naming is deliberately NON-standard and was REVERSED 2026-07-21:
  **XY plane (z-normal, y vertical) = "Top view"; XZ plane (y-normal, z vertical) = "Side view."**
  x (propagation) always horizontal; ux always horizontal in far-field plots.
  (CLAUDE.md §8 still states the OLD convention — flag it, do not edit without permission.)
- Devices are named by GEOMETRY, never "champion" (e.g. "rect-1050", "W1050 + see-saw").
(project_it11_device_naming, project_visualization_conventions, project_tm_loss_new_physics_round)

---

## 2. Closed studies — the verdict table

### 2.1 Trench / air-trench

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Air trench (phase-3 of the BIC/Kerker batch) | lateral AIR (n=1.0) trenches parallel to the guide, TIR the near-axial leak (79° > 43.8° crit) → shrink the in-plane light cone | **POSITIVE — the one real novel win of that program** | stack + air-trench (L=84 µm, W=800 nm, d=1.8 µm): loss **0.0423**, T 0.9573, fwhm **15.30 µm** (narrower), Q 1489, R≈0, single resonance, vs stack 0.0545 / 0.9449 / 15.45 → **−22 % loss at fixed/narrower width ⇒ ESCAPES the loss-vs-width Pareto**. Jitter twin d=1.825 → 0.0424. Clean d-optimum (1.2 worse → 1.8 best → 2.4 fading). SiN strip at the same spot CATASTROPHIC (loss 0.40, mode 27 µm) ⇒ low-index/TIR mechanism load-bearing. Short 20 µm trench HURTS (0.092, end-scatter) ⇒ must span the full arm | job **118893**, `results_from_athena/tm_air_trench/` (+ `layout_stack_plus_airtrench_d1p8.fsp`) | project_bic_kerker_batch1_dispatch |
| Air-trench theory ceiling (zero-GPU) | anisotropic / low-index SWG cladding vs the trench | **CLOSED — same mechanism, refined form** | n_eff 1.30–1.35 SWG cuts ~39 % of the edge-piled lateral leak → Δloss ≈ −0.012, matching the trench. Air-TIR reflects only the grazing 46 % (|kx|/kc > 0.69) | docs/theory_gate_supercavity_aniso_2026-07-07.md | project_bic_kerker_batch1_dispatch |
| USER STANCE on the air trench | — | unimpressed: "just putting air in the cladding" = the known low-index/suspension lever | — | — | project_bic_kerker_batch1_dispatch |
| Trench shape / flare — d(x) axis | shaped (flared) trench wall vs straight d=1.8 µm, on apod-10 | **CLOSED NEGATIVE, twice, two clusters** | ctrl 0.9772 / loss 0.0226 · straight d1800 0.9811 / 0.0187 · **FLARE 0.9293 / 0.0689** · jitter 0.9279 (pair spread 0.0014 < floor). Prediction was +0.001..0.002; measured **−0.052 vs straight** (~25× floor). Round 2 SMOOTH flare (64 nm facets) 0.9330 — recovers only +0.0037 (~7 %) ⇒ penalty is the LPG **envelope**, not a staircase artifact. Mechanism: any wall modulation with period 3.1–24 µm phase-matches the core (n_eff 1.508) → accidental long-period grating; C_cav ≈ 230. Safe periods only <0.53 µm (SWG) or ≫24 µm. **Do NOT re-propose shaped/curved trench walls** | Athena **126913**; IGUM **47357** (+ **47364**); `results_from_athena/trench_flare_apod/`, `results_from_igum/{trench_flare_apod,trench_apod20}/` | reference_air_trench_formulation_doc |
| apod-20 + full-z trench | do the two stack? | **NULL** | apod20 ctrl 0.9827 (matches accurate 0.9829) vs +full-z trench 0.9831 → dT **+0.0004 < 0.0018 floor**. Trench gain vs apod depth: uniform +0.0159 / apod10 +0.0039 / apod20 ~0 = the leak-budget catch-22 measured end-to-end. **BEST SIMPLE DEVICE = apod-20 alone (T 0.983, loss ~0.017)** | IGUM **47364**, `results_from_igum/trench_apod20/` | reference_air_trench_formulation_doc |
| TM −3 dB / 20 µm trench vs no-trench | max Q_L at peak T = −3 dB for a 20 µm TM mode | **CLOSED POSITIVE (+35 %)** | corr locked **325 nm** (fit corr(20 µm)=324.7, resid 0.17 µm). no-trench **N=165, T 0.491 (−3.09 dB), Q_L 13,930, fwhm 19.97 µm, λ 1559.00**; full-z trench (W800/d1800/H12000, L 178 µm) **N=170, T 0.5021 (−2.99 dB), Q_L 18,777, fwhm 19.57 µm, λ 1558.27**; N=169 bracket T 0.513 / Q 18,279. Q_i 46.5k vs 64.2k at N=165, Q_c ~unchanged (~11 %). Q_i ∝ corr^−2.9 (266–400) | IGUM **47910/48458/48711/48973**, 29 sims, `results_from_igum/trench_q3db_20um/results/` | project_trench_q3db_20um_closed |
| Flush-top trench | trench top flush with SiN top (z −3.975 → +0.175 µm), oxide below the floor — single-litho-compatible | **POSITIVE, ~62 % of the full-z advantage** | N80/corr400 single run: T 0.8996 (−0.460 dB), λ 1558.066, Q 1392, fwhm 15.47 µm (keeps 77 % of the full-z dB gain, +0.065 of +0.085). Final fixed-mesh family: **flush N168 T 0.5017 (−2.996 dB) λ 1558.482 Q 16,942 fwhm 19.68 µm (+21.6 %)**; bracket N169 T 0.4912 (−3.087 dB) Q 17,392. At IDENTICAL numerics: ctrl N165 Q 13,930 | flush N168 Q 16,942 | full-z N170 Q 18,777 | Athena **128918**, **128925**, **129103**, **129105_1**, **129730_2**; ctrl IGUM **50733**; `results_from_athena/{trench_flush_top,trench_flush_q3db}/`, PPTX `trench_normalized_cross_sections.pptx/.png` | project_trench_flush_top_study |
| ★z-symmetry numerics discovery (inside the flush study) | `use_z_symmetry=False` (+ symmetric z mesh) inertness | **NOT inert at corr 325** | corr325/N165 z-sym OFF vs stored ON: **λ +2.10 nm, T −0.21 dB, Q +7 %** — while at corr400/N80 the same change was null (5 pm). Any z-sym-OFF study MUST carry its own matched control. Artifact-family rows archived in `results_zmesh179_artifact/` | — | project_trench_flush_top_study |
| Trench height scan, N=150 W800 | 12 heights 350 nm → full-z, to fill the knee | **CLOSED — smooth saturating rise** | h (nm) → T / Q: 0 (ctrl) 0.1842/14.2k · 350 0.1999/15.4k · 450 0.2013 · 575 0.2033 · 735 0.2055 · 940 0.2080 · 1200 0.2114 · 1550 0.2147 · 2000 0.2185/17.0k · 2550 0.2226 · 3250 0.2280 · 4000 0.2287 · 5400 0.2315 · 6900 0.2322 · full-z 0.2315/17.7k. Half the gain by h≈2 µm, ~95 % by 4 µm, flat above 5.4 µm; λ_res pins at 1557.844 above h≈3.2 µm. **Cross-cluster equivalence ESTABLISHED** (IGUM ctrl = Athena ctrl to 0.0001) | IGUM **41767** + **42317** + **42325** (15/15), `results_from_igum/trench_n150_hscan/` + `trench_hscan_T_Q.{png,fig}` | project_trench_n150_hscan_igum |
| Trench + TE | is the trench a TE lever? | **NULL for TE** | `trench_te_apod` ctrl 0.8756 vs 0.8747 | — | project_te_q3db_20um |

### 2.2 In-core holes

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Single-hole position scan (`tm_hole_scan`) | one SiO2 cylinder r=100 nm on axis, x = k·Λ/8, k=0..96, + r=0 ctrl | **CLOSED — parasitic-to-neutral, never beneficial** | worst T 0.828→0.724 at x≈0.65 µm; at favorable intra-period positions nearly free. Always BLUE-shifts λ_res (up to −0.45 nm near the cavity) ⇒ possible post-fab λ-trim knob. Baseline (default 3.8 µm domain) λ 1558.576, T 0.8278, Q 1284, loss 0.164 | 98/98, job **116152** (+ **116272** rescue of tasks 13–97), `results_from_athena/tm_scatterer_scan/FINDINGS.md` | project_tm_scatterer_scan |
| SiO2 in-core hole LATTICE | r=100 nm SiO2 at y=0, one per narrow-tooth centre, all 160 periods, anchored TM (pitch 516.83, corr 400, h350) | **CLOSED — every variant strongly harmful** | own ctrl T 0.825 / loss 0.166 / λ 1558.56 / fwhm 15.5 µm. matched corr400: λ 1547.8, **T 0.032** (jitter twin 0.040) — defect peak nearly annihilated; corr300 trim: 1549.0, T 0.160, loss 0.49 (3× ctrl); period-detuned 545 nm: T 0.340, loss 0.52 (worst); lattice shifted +pitch/4: T 0.409, loss 0.47. All blue-shift λ ~−10 nm. Jitter floor 0.0002 (passband) / ~0.008 (collapsed peak) ⇒ effects 50–1000× the floor. **TRAP: the stored resonance fields 1571.5 / T 0.911 / fwhm_m 63 µm are a finder mis-pick of the passband** | job **123303** (6/6), `results_from_athena/tm_hole_lattice/` | project_hole_lattice_closed |
| In-core SiO2 hole COMB (Λ524 / 270° / 31 holes) | r 30–110 hole combs on the short TM corr-400 N=80 device, then the equal-width discriminator | **CLOSED NEGATIVE 2026-09-15 — gain was only the corr→width lever** | vs plain ctrl T 0.8851 / 15.53 µm / Q_i 22.4k: r30 0.9008/16.4 µm · r40 0.9113/17.2 · r50 0.9244/18.5 · r80 0.9280/23.0 · r110 0.8869/28.2 (dT/dwidth ≈ **+0.018/µm** = the plain corrugation ladder's own slope). Equal-width test r50+corr477 (width 15.39 µm): T 0.7922, Q_L 2102, **Q_i 19.1k (−15 %)**, Q_c +67 %. Equal-width #2: r60+corr465 Q_i −9 %, r80+corr503 −23 %; −3 dB Q estimates at 15.5 µm: plain Q_L 6627 vs 6117 / 5031 (pair r50 5715). Best case at the hole's own optimum Λ527: r50@456 → Q_i −1.5 %, −3 dB Q **0.0 %** = NEUTRAL, never a gain. **Do not propose in-core holes again for the fixed-width spec** | Athena **148812/149355/149982/150391/150429/150458/150488** (+ **151476**), `runners/scatterers/COMB_HANDOFF.md` (in-core row) | project_incore_hole_comb_closed |
| useful by-product | — | the q3db TM width knob (1/w linear in corr) **transferred to the decorated device to 1 %** — a width retune of a decorated device is a one-run job | — | — | project_incore_hole_comb_closed |

### 2.3 Cladding reflectors

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Cladding-reflector study | 1D SiN Bragg mirror (DBR, quarter-wave SiN strips in y, period 524 nm ≈ device pitch / grazing 608 nm, N=5/8/10, d=1.5/1.8, full 83.74 µm length) + 2D square SiN-rod PhC (r 105–140 nm, central ±25 µm, a = 460/500/580 nm) + distance scan d=1.2/1.5/1.8/2.1 + jitter + short-±12 µm control | **CLOSED NEGATIVE — every SiN lateral reflector loses** | theory ceiling from the measured leak spectrum: air-TIR reflects the grazing 46 % (−0.012); tuned 1D DBR 60–65 % (complementary, near-normal); a COMPLETE 2D gap 100 % (loss 0.0545 → ~0.023). Reflected flux is an UPPER bound — recoupling is kx-weighted toward grazing, which the trench already catches. Base = the stack (loss 0.0545), TM, accurate mesh, box y=16 µm, window 1556.5/40/3001 | jobs **119163** and **119194** CANCELLED (128 G, doomed by OOM) → **119208**, 15 tasks, all at 256 G; runner `runners/sweeps/tm_cladding_reflector.py` | project_cladding_reflector_dispatch (verdict line from MEMORY.md index) |
| OOM root cause found here | — | **infra fix, keep** | the SOLVE is fine (52 min); post-processing (`get_s_and_t_matrix` port mode-expansion over the 16 µm box) spikes HOST RAM to **132.8 G**. Two bugs fixed: `SBATCH_MEM` was wired only into the `--option2` branch (now also `--option3`, `deploy_athena.sh` ~line 1166); QOS `24h_1g`/`4d_1g` cap per-job memory at **275 G** so 300 G is rejected — use **256 G** | — | project_cladding_reflector_dispatch |
| Silicon as reflector material (user Q) | — | **parked platform change, not run** | Si is bad as a solid trench (n=3.48, no TIR oxide→Si, severe parasitic guide — worse than the SiN-strip control at loss 0.40) but would be EXCELLENT as DBR/PhC material (contrast 2.41 → complete 2D gap, approaching the −0.032 ceiling) | — | project_cladding_reflector_dispatch |
| SiN strip reflectors near/along the arms | 24 rows of strips on the stack base | **CLOSED — all 23/24 variants HURT** | d=1.2 µm full-arm strips destructive (loss 0.100/0.120 vs stack 0.0545) with λ_res dragged +6–8 nm = coupling/drain regime; all full-arm strips +4 to +137e-3, damage decreases with d, NO recycling oscillation; near-cavity worst **0.242** | job **118360** (box Ybox7p6 — tags differ) | project_tm_loss_new_physics_round |

### 2.4 BIC, Kerker and counterdiabatic

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Phase-0 theory gates (zero GPU) | 4 routes on paper | mixed | 4.1a symmetry-protected BIC **FAIL/dropped** (in-cone radiation measured 100 % y-even; the only protecting mirror also kills the y-even port → T=0). 4.1b FW-BIC **PASS conditional** (lossy-partner regime g2 = 2–15 nm ≫ g1 = 0.031 nm → 5.3× radiative suppression, partner 16 nm out, admixture 0.5 %; needs pattern overlap ρ ≥ 0.82). 4.1c vertical anti-phase 2Λ = the only planar handle on the 38 % vertical share. 4.2 **backward-Kerker IMPOSSIBLE** at m=1.364 (best B:F 0.95:1); forward-Huygens strong (TM ~65:1 @ r=250 nm, TE ~10⁵:1 @ r=260 nm) but placement-map ceiling 1 pair ≤ +0.008 T, 2 pairs ≤ +0.016. 4.3 counterdiabatic MARGINAL (predicts −2.9 % ≈ the uniform pair's −2.8 %) | `docs/phase0_gate_verdicts_2026-07-06.md`, `python_tools/phase0_*.py` | project_bic_scatterer_program |
| Batch 1 (35 tasks) | Huygens/Kerker scatterers (TM+TE), vertical 2Λ alternation, counterdiabatic falsifier | **Huygens DEAD; 2Λ NULL; CD initially "winner"** | Huygens/Kerker: ALL rows raised loss, bigger/directional radii WORSE (passive Mie phase ≠ optimal cancel phase); the old +0.0026 anchor does not reproduce near the cavity. Vertical 2Λ: best −0.0003 (inside the opt jitter floor), real ±sign asymmetry = coherent channel, too small to use. CD: loss 0.0545→0.0504 (−7.5 %) at +0.4 % fwhm, monotonic + sign-dependent (scale +2 worsens to 0.0627) | job **118618**, `results_from_athena/tm_bic_kerker_batch1/` + FINDINGS.md + counterdiabatic_quicklook.png | project_bic_kerker_batch1_dispatch |
| FW-BIC (side-coupled twin cavity) | 20-row LOCATE grid, device 2 passive detuned partner (new knob `avg_corrugation_width_2_m`) | **FAILED** | every side-coupled cell RAISED device-1 loss (**0.23–0.47** vs the 0.112 weak-coupling ref, peak T down to 0.4); more detuning reduces the damage but NEVER dips below the isolated 0.077. The partner is a lossy sink (ρ too low) | job **118734** (verdict from 10/20 cells), `results_from_athena/tm_fw_bic_scan/` | project_bic_kerker_batch1_dispatch |
| Batch-1c CD profile scan (13 rows, accurate) | CD amplitude {0..−6} + DISTRIBUTION controls (uniform-14, lumped-2 at matched total shift) | **DECISIVE — CD is NOT special; it is a loss-vs-width Pareto** | at MATCHED total shift (fwhm µm, loss): lumped-2 (15.47, 0.053) → CD-shape (15.51, 0.050) → **uniform-14 (15.99, 0.040)**. More distributed = lower loss BUT wider mode. CD's promise (loss cut WITHOUT width cost) FALSIFIED. Within ±1 % width the best is CD scale −3: loss ~0.0489 (−9 % vs stack 0.0541) at fwhm +0.7 %. Relax to +5 % width (16.21 µm) → loss 0.0374 (−31 %). Also: at matched fwhm ~16 µm UNIFORM (0.0397) beats the CD quadrature profile (0.0467) | job **118809**, `results_from_athena/tm_cd_profile_scan/` + `cd_pareto_control.png`, FINDINGS.md (+CORRECTION) | project_bic_kerker_batch1_dispatch, project_innermost_tooth_recycling_theory |
| ★CORRECTION on the record | — | the earlier "counterdiabatic winner / −31 % / supersedes stack" was **OVER-STATED** — s3 uniform points were mislabeled in the first quick-look; −31.5 % is the uniform control at **+4.9 % width**, NOT fixed width | — | — | project_innermost_tooth_recycling_theory |
| Phase-2 scatterer closeout (9 rows, accurate) | real-α SiN post at the correct site (500,950), air-void α<0 at (120,2000), CD×2Λ combo | **closes the scatterer route** | SiN posts DEAD (add loss, worse with radius). Air-void α<0: loss 0.0535 vs ctrl 0.0541 (**−0.0006**) while a SiN post at the SAME site = 0.0562 → **sign flip vindicates the Green's-function theory but magnitude ~10× below model** (parasitic-limited). CD×2Λ null | job **118865**, `results_from_athena/tm_novel_phase2/` | project_bic_kerker_batch1_dispatch |
| Single-cavity supercavity / Friedrich–Wintgen (TIER 1) | zero-GPU CMT theory gate | **MIRAGE — closed on paper, no GPU spent** | overlap is NOT the blocker (even 2-node partner ρ = 0.88, clears the 0.82 gate; floor (1−ρ²)·loss0 = 0.012). It dies structurally: a single π-defect has NO 2nd co-located gap mode (higher states expelled to bands; forcing one splits the resonance or widens the mode), and a TRANSMISSION port cannot harvest a BIC | `docs/theory_gate_supercavity_aniso_2026-07-07.md`, `python_tools/phase0_supercavity_fw.py` | project_bic_kerker_batch1_dispatch |
| Novelty verdict (docs/novelty_analysis_2026-07-07.md) | — | **standing conclusion** | for a SINGLE resonance at FIXED width, loss = the mode envelope's light-cone Fourier tail. Tooth-width + tooth-shift span the FULL coupled-mode envelope equation ⇒ envelope optimization = the inverse-design space ⇒ CD and everything planar/fixed-width is inverse-design-reachable. **Mirror-symmetric is PROVABLY optimal** (antisym perturbation → odd δA ⊥ even A0 → strictly adds radiation). Only cladding engineering (−0.012..−0.015 ceiling) and the envelope Pareto survive; true novelty requires relaxing a constraint (width / back-reflector / platform / active) | — | project_bic_kerker_batch1_dispatch |
| Dropped in this program | — | — | sym-BIC (4.1a), backward-Kerker (impossible), Huygens, vertical 2Λ. Two-defect / supermode routes DE-PRIORITIZED by the **single-resonance constraint** (two separated π-shifts split into an observed doublet) | — | project_bic_scatterer_program |

### 2.5 Apodization

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Apodization sweep TE vs TM, linear | `n_apod_periods_each_side` = {2,5,10,20} × {TE,TM}, `apod_method="linear"`, `center_mod_depth_nm=4.0`, N=80, 150 nm window | done | 0-teeth baselines: TE λ 1570.7, T 0.86, fwhm_m 15.2 µm; TM λ 1523.6, T 0.97, fwhm 17.9 µm (0-teeth points REUSED from `results_from_athena/run_tm_vs_te/results/`) | Athena array **96506** (8 tasks), `results_from_athena/tm_te_apod/`, plot `matlab_plotting/plot_apodization_vs.m` | project_apodization_sweep_tm_te |
| tanh sibling | identical, `apod_method="tanh"`, `tanh_steepness=2.0` | done | — | Athena **97137**, `results_from_athena/tm_te_apod_tanh/` | project_apodization_sweep_tm_te |
| TM at pitch 518.3, linear vs tanh | TM re-run at the matched wavelength, n_apod {2,5,10,20} × {linear,tanh} | **COMPLETE — linear → higher T, tanh → tighter mode** | teeth 0 = 0.9584 / 19.03 µm; linear 2 = 0.9718/20.56, 5 = 0.9795/21.97, 10 = 0.9836/24.16, 20 = 0.9849/28.64; tanh 2 = 0.9667/20.01, 5 = 0.9717/20.44, 10 = 0.9773/21.24, 20 = 0.9818/22.81. Pitch-518.3 peak T (~0.958) < pitch-500 (~0.974) because peak T is radiation-limited and shifts with pitch/λ. 4 nm mod-depth floor chosen over 10 nm | Athena **97304** (+ **97355** A20 recovery after the shared-`sweep_list.txt` race), `matlab_plotting/plot_apod_tm518_lin_vs_tanh_headless.m` | project_apodization_sweep_tm_te |
| Apodization vs the defect-local family (Pareto) | apod ladder against the stack family at equal width | **apodization LOSES below ~+3 % width; crossover ~+10 %; modularity SIGN-INVERTS** | apod n5 loss 0.0416 at +14.8 % fwhm, n10 0.0229 at +25.9 %, n20 0.0169 at +48.7 %. Defect-local family dominates to ~+3 % (triple 20×3: 0.0403 at +2.9 % beats apod n5). **The pair goes −27.4e-3 standalone → +26.2e-3 under apod10** (width-null inverts too) ⇒ apodization and defect-local corrections are the SAME resource (interface impedance matching). apod10 + full stack 0.0349 at +16.6 % still beats pure apod at equal width ⇒ a combined design must **CO-OPTIMIZE, not compose** | job **118293** (12/12), `results_from_athena/tm_pareto_stack_vs_apod/` + `pareto_stack_vs_apod.png` | project_tm_loss_new_physics_round |
| Combs vs apodization | — | **never stack** | (MEMORY.md index, comb program) | — | project_comb_physics_rethink (index) |

### 2.6 The cavity-loss program

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Cavity loss program round 1 (`anti_moment_cavity`, 13 tasks, ALL accurate mesh dx≈35, converged box, window 1558.5/40/3001, λ_res 1556.1) | cavity width ladder + inner "see-saw" (teeth ±1 = 1000+δ, ±2 = 1000−δ, zero net area, even parity) | **POSITIVE and CLOSED** | control loss 0.1174. Width ladder: 1000→0.0839, 1025→0.0829, **1050→0.0823**, 1052→0.0821 (= in-study jitter floor 2e-4), 1075→0.0823, 1100→0.0830 ⇒ FLAT plateau 1040–1075, width knob exhausted. See-saw: δ=+10→0.0814, **+20→0.0810**, +30→0.0810 (saturates); −10→0.0834, −20→0.0851, −30→0.0871 ⇒ antisymmetric + saturating + linear-through-zero = genuine interference cancellation. **BEST: rect-1050 + see-saw δ=+20 (teeth ±1=1020 / ±2=980) → loss 0.0810 (−31 % vs control), T 0.878→0.9179, fwhm +0.8 %, λ_res unmoved.** Plain rect-1050 fallback: −29.9 % ACC-confirmed (job 117784: 0.1174→0.0823, T 0.917, fwhm +0.62 %) | job **117814**, figures `results_from_athena/anti_moment_cavity/anti_moment_cavity_summary.png/.fig`, script `matlab_plotting/plot_anti_moment_cavity.m`; jobs ledger 116979 117000 117042 117054 117063 117434 117486 117500 117508 117530 117553 117784 | project_loss_exploration_chain |
| Cavity SHAPES on top of rect-1050 | barrel/hourglass/hann/gauss/tri3/5/7/dbl2/sldn/slup/tilt | **NULL to harmful — the cavity optimum is purely SCALAR (added dielectric area)** | barrel/tri7 HURT (area past optimum); optimum reverses by 1400 (+7 %). Tilt on 1050 REAL but tiny: −0.00074/−0.00075 for depth 150/300 (identical ⇒ saturated), ~7× the 1e-4 accurate floor = −0.9 % relative → fab the plain rectangle. Equal-area Hann ties rect-1050, more-area Hann worse | combo job **117553**; `cavity_hann_sweep` job **118529** | project_loss_exploration_chain, project_bic_scatterer_program |
| Distributed π-shift (round 1 form) | spread the π slip over teeth | **FALSIFIED in that form** | ALL variants +21..+39 % loss and fwhm also widens — each shifted gap is its own radiating kink; lumped shift optimal | job **117530** | project_loss_exploration_chain |
| K-space diagnostic (Round 7) | where the radiating weight lives | **the program-defining measurement** | only ~**30 %** of radiating weight is cavity-local and the best device already harvests ≈ that; the remaining ~**70 % is distributed along the arms**. Δk·σ = 1.70; ~15/23/29 % of radiating weight within ±1/2/3 pitches. GOTCHA: the Lumerical monitor x-grid is NON-UNIFORM — resample before any FFT (first pass was wrong by ×1.48 in k) | `results_from_athena/radiation_kspace_diag/`, `matlab_plotting/plot_radiation_kspace_diag.m` | project_loss_exploration_chain, reference_loss_reduction_options |
| Round-2 polarimetry (7 rows) | in-plane vs vertical split, lobe angle, TM→TE conversion | **gating diagnostic, closes the lateral-leakage route** | audit closes 99–100 %; **in-plane 62 % / vertical 38 %** (mesh-robust); **f_TE ≈ 0 — zero polarization conversion, the TM→TE lateral-leakage route is measured DEAD**; both planes NEAR-AXIAL (|ux| ≈ 0.98); cavity width controls the broadside pedestal (mean |ux| 0.52 → 0.77 rect-1050 → 0.44 W1400); rect-1050 cut 2·side 0.068→0.034; 27/43/77 % of side radiation within ±3 pitches/±6/±12 µm | job **117907**, `results_from_athena/tm_radiation_polarimetry/FINDINGS.md` | project_tm_loss_new_physics_round |
| Centre completion (49 rows accurate) | cavity L × W, W1600/W2100 two-edge discriminators, tooth-shift retest, ptw × detuning additivity | **new best; two-edge model FALSIFIED** | **W1050 + inner gap-shift pair [+20,+20] (lengthen_cavity ON): loss 0.0549, T 0.9444, fwhm_m +1.0 % (at bound), Q 1403, λ 1556.58** — ΔT +0.059 vs the W800 baseline, +0.028 vs rect-1050. W1600 +30 %, W2100 +53 %, monotonic ⇒ the cavity-width optimum is a local **moment null** (Johnson-type), NOT two-edge interference. Cavity LENGTHENING (det −40) cuts loss −16e-3 but fwhm +4.2 % ⇒ PARKED (delocalization in disguise). See-saw plane best (40,−20) −1.84e-3; tooth-3 ≈ nothing; narrow see-saw HURTS | job **117927** (49/49), `results_from_athena/tm_center_completion/FINDINGS.md` + `center_completion_summary.png` | project_tm_loss_new_physics_round |
| Shift frontier (20 rows) | singles/pairs/triples/length at many doses | **the frontier law** | the tradeoff is LINEAR, ≈ **−1e-3 loss per +0.1 % fwhm**, SAME slope for singles/pairs/triples/length ⇒ one dose parameter. **Best in-bound = "the stack": W1050 + pair[+20,+20] + see-saw(1040,980) → loss 0.0545, T 0.9449, fwhm +0.9 %** (bare pair 0.0549 ≈ same; see-saw adds only −0.4e-3 ON the pair ⇒ additivity breaks within-type). Off-bound: triple 20×3 → 0.0403, T 0.9591 at +2.9 % | job **118214** (20/20), `results_from_athena/tm_shift_frontier/FINDINGS.md` + `shift_frontier_summary.png`; best .fsp `…/tm_shift_frontier/layouts/layout_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8.fsp` | project_tm_loss_new_physics_round |
| Derived-boundary-profile test (7 rows accurate) | a per-tooth profile derived from a calibrated boundary-perturbation kernel (cavity −18.8; teeth [−12.3, +17.8, +9.8]; gaps [+18.7, −13.0, −12.3] nm), ×0.5/×1/×2 and sign-flipped | **ROUND CLOSED NEGATIVE — the stack is a genuine LOCAL OPTIMUM** | the derived profile WORSENS the stack at every amplitude AND in the sign-flipped direction (**+4.4…+7.3e-3**, jitter-solid, fwhm in bound) ⇒ **the −1e-3-per-+0.1 %-fwhm frontier is fundamental for local perturbations at fixed mode width.** Residual 0.0545 = envelope/arm physics (~55 % near-axial in-plane + ~45 % vertical). Without a fwhm guard the "optimal" profile fakes −11 % via a smooth taper (delocalization manifold); with the x²-moment guard it collapses to −1.8 % | job **118473**, `results_from_athena/tm_derived_profile/FINDINGS.md`; kernels `python_tools/derive_boundary_profile{,_stack}.py` (sign/structure tool, NOT an optimizer); field export job **118462** | project_tm_loss_new_physics_round |
| Exotic innermost-tooth SHAPE | notch / step / wedge_cav on the innermost tooth pair vs rect control | **CLOSED — rides the width Pareto, no special recycling** | rect control loss 0.1106. notch **−0.0048** (0.1058, T +0.005) but at **+2.8 % wider mode and UNCHANGED Q (1306 ≈ 1304)**; step +0.0044 (worse); wedge_cav −0.033 T MUCH worse (a single tilt does not retroreflect the grazing leak). Theory: at fixed area all in-plane shapes change radiated power by **<0.2 %** vs rect (tooth 258 nm but radiation couples only to features ≥1 µm = 2π/kc; and the β-shift samples ã at kx−β ≈ −Δk ≈ 0, i.e. the tooth's DC/AREA). Ceiling for innermost-teeth-only: ΔT **+0.001..+0.006** (~15 % of the in-cone leak within ±1 tooth, ~29 % within ±3). CAUTION: an intermediate "edge-split −19 %" result was a PHYSICS ERROR (ã evaluated at unshifted kx) | job **119539**, runner `runners/sweeps/tm_exotic_recycle.py`, `docs/theory_innermost_recycling_2026-07-08.{md,pdf}`, `docs/tm_exotic_recycle_119539_2026-07-08.png` | project_innermost_tooth_recycling_theory |
| SSH gap dimerization | 1D TMM, sign-validated on the distributed-shift failure | **KILLED ON PAPER, no GPU spent** | every row increases light-cone weight | `docs/tm_loss_program_phase0_2026-07-05.md` | project_tm_loss_new_physics_round |
| shift_target narrow vs wide (corr-400 TM, N=80) | should the shift shorten the NARROW or the WIDE segment? | **wide-target helps but is STRICTLY DOMINATED — axis not reopened** | s=0 ctrl λ 1558.617 / T 0.8864 / Q 1311.4 / 15.532 µm. narrow +51.68 → 0.9038 (+0.0174) / 15.707 (+1.12 %); narrow +103.37 → 0.9179 (+0.0315) / 16.260 (+4.69 %); wide +51.68 → 0.9006 (+0.0142) / 15.783 (+1.62 %); wide +103.37 → 0.9067 (+0.0203) / 16.404 (+5.61 %); neg −51.68 → 0.8665 (−0.0199); neg −103.37 → 0.8457 (−0.0407). dT per % width: narrow 0.0155/0.0067 vs wide 0.0088/0.0036 (**~1.8×**). The +103 pair differs by 0.0112 ≈ 6× the 0.0018 jitter floor (solid); the +51 pair only 1.8× (not independently decisive). **Neither sign NARROWS the mode** (envelope ~exp(−∫q dx), q = √(κ²−δ²) ≤ κ ⇒ detuning can only widen) | jobs **134977** / **134984**, `results_from_athena/tm_shift_c400/results/` (tags `_S52w`/`_S103w` wide, `dsh1Sm52`/`dsh1Sm103` negative), anchor in `asym_dw_study/results/` | project_shift_target_sign_test |
| ★prediction trap recorded there | — | — | the cavity absorbs 2Σs whichever segment is shortened ⇒ **cavity lengthening is COMMON-MODE and is the dominant T lever**; the duty-cycle/⟨n_eff⟩ term is only the DIFFERENTIAL. λ rises +0.569 nm (narrow) vs +0.259 nm (wide) at s=103.37 | — | project_shift_target_sign_test |
| OUT-OF-SCOPE parking list (do NOT recommend at fixed width) | — | — | W1000/C500 + cav1250 −60 % at +6 % fwhm; tapered island (8 teeth) −36 % at +4.8 %; TE whole-device sinusoid −29 % at +7.5 %; TM sinusoid −10 % at +0.8 %; TE barrel300 −9 % | — | project_loss_exploration_chain |
| USER SCOPE for this program (violated twice, keep honoring) | — | — | fixed GIVEN device pitch 516.83 / corr 400 / W800 / h350 / TM / n 1.97-1.444; modify ONLY the cavity segment + at most 1–2 adjacent teeth; **fwhm change ≤ ~1 %** (+5 % "is quite a lot"); NO corrugation changes, NO core-width changes, NO tapers | — | project_loss_exploration_chain |

### 2.7 Scatterers and PSO

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Lateral pillar-pair recycling scan (187 tasks) | SiN cylinder pairs at (x, ±1.0 µm), r=150 x=0..12.15 µm step 135 nm (91 rows) + r=100/r=200 step 270 nm (46+46) + 3 jitter (+25 nm) + control | **REAL but SMALL — route closed at a ~+0.003 ceiling** | control (task 0): λ 1558.566, T 0.799, Q 1267, loss 1−R−T = **0.189** (corr-400 TM loss ~19 %, far above paper_8's 4 % corr-300 figure). Accurate-mesh confirm (dx=35, job **116190**): **ΔT = +0.0021, Δloss = −0.0023, ΔQ = +1.6 at r=100 nm, x=0.81 µm**, with the jitter spread collapsing **0.0018 → 0.0001** (>20× significance) — this two-step is the template for near-floor effects. Worst case r200@1.62 µm ΔT −0.012..−0.065 also real. ~1 % of radiated power recycled per pair; the optimum is a ≥25 nm-wide plateau; **r=200 NEVER improves** (self-scattering wins). Accurate-mesh absolutes shift (λ 1558.57→1555.90, T 0.80→0.77) — compare within-mesh only | **115787** (ctrl), **115895** (tasks 1–100), chunk 2 (101–186); demo field maps **116169**; `results_from_athena/tm_scatterer_scan/FINDINGS.md` | project_tm_scatterer_scan |
| Scatterer radius ladder | r = {80, 100, 125} at x=810, accurate mesh, converged box | finite optimum ≈ 100 nm | dT = +0.0020 / +0.0026 / +0.0018 → r=80 explored, slightly worse than 100 | job **116896** | project_loss_exploration_chain |
| Narrow-touch fused-pillar device | r=80 nm n=1.97 cylinder pairs FUSED to the body at two sites: (x=0, y=±480) fused to the cavity (half-width 400) and (x=270, y=±380) fused to the narrow-section sidewall | **SAVED CANDIDATE — best TM corr-400 result to date at optimization mesh** | **T = 0.9310 (+0.0448 vs control 0.8862), resonant loss 0.0672 (−39 % vs 0.1100), Q 1352 (vs 1319), λ_res 1558.90 (+0.29 nm), spatial FWHM 16.02 µm (+3.2 % vs 15.53), spectral FWHM 1.153 nm (−2.4 %)**. Beats: floating pair [0,270]@700 +0.0227; descent best pair@480 +0.0377; plain rect-1050 +0.0354. **CAVEAT: single point at optimization mesh (dy ~46–50 nm, curves staircased) — needs accurate-mesh (ideally dy=25) confirm before quoting as final.** Mechanism likely local width-profile engineering near the cavity, not circle-specific; the rectangle-equivalent discriminator was proposed, never run | job **121843**; `results_from_athena/scat_e_validate/results/result_N80_TM_W800_Ybox6p8_Zbox8p8_scR80_arr2_X0to270_Y480to380_pair_ff.mat`; layout `…/scat_e_validate/layout_narrow_touch_X0to270_Y480to380_LOCAL.fsp`; runner `runners/scatterers/scat_e_validate.py` (ROUND 8) | project_narrow_touch_design |
| ★TRAP | — | — | `tm_scatterer_scan.build_base()` returns `scatterer.enabled=True` with the DEFAULT radius 150 nm. Any runner reusing it for a NON-scatterer study MUST set `BASE.scatterer.enabled = False` (tm_width_lightline forgot → job **116970** ran a spurious r=150 pillar pair in every row; cancelled, clean redispatch **116974**). Verify via the stored `scatterer_r_m`/`scatterer_n_sites` in the .mat. Also: a single non-mirrored off-axis scatterer + y-symmetry ON would silently simulate a pair — `bragg_device` now RAISES. MATLAB `r_nm == 150` fails on the 150e-9·1e9 round-trip — `round()` first | — | project_tm_scatterer_scan |
| TM 3-parameter PSO (DW1/DW2/cavity, no shift) | gradient-free transmission maximization, N=80, pitch 518.3 | **completed with a modest ~1–2 % lift; headline Δ was an unfair comparison** | first attempt job **97162** COMPLETED but **INVALID** — the parametric `.fsp` builder produces a DEAD TM device (modal |S21|² ≈ 0.000829 flat for EVERY particle, including a geometrically-baseline one that the normal builder reads at T = 0.945). FIX = `rebuild_per_particle=True` (skip the parametric .fsp, score each particle through the full `run_single_sim` path); VALIDATED job **97225** (gen0-particle1 = 0.9454 vs baseline 0.9451). The whole parametric/FOM-.lsf/static-skeleton layer is adjoint-only scaffolding. Index correction 1.977 → 1.9963 (**97299 → 97316**): baseline peak_T 0.9582 @ 1571.4 nm. **FINAL (97316, converged gen 3, 3 h 23 m): DW1 = 95.2, DW2 = 102.0, cavity = 854.3, shifts 0; coarse peak_T 0.9747 (+0.016 coarse-vs-coarse); accurate-mesh verified true_peak_T 0.9561 @ 1568.3 nm.** The driver's headline Δ = −0.002 is an UNFAIR coarse-baseline-vs-accurate-optimum comparison (accurate reads ~0.018 below coarse) — a true gain needs the baseline re-run at accurate mesh, still OPEN | `runners/gradient_free_design/optimize_transmission_tm.py` | project_tm_transmission_pso |
| 5-param PSO follow-up (adds the two shift slots) | does shift help TM on top of the 3-param optimum? | dispatched; smart seed = the 3-param optimum | job **97635**, pop 15 × 10 gens (~165 evals), label `transmission_gf_tm_shift`. Later reconciliation: that run used shifts of **118/138 nm** and reached T = 0.9618 — "shift adds little for TM" was a statement about marginal value on top of free DW1/DW2+cavity at the OLD geometry, not about fixed-corr-400 | — | project_tm_transmission_pso, project_tm_loss_new_physics_round |
| Literature route ranking (TM loss) | ranked, citation-backed options | reference only | true corr-400 radiated budget ≈ 10 % of T post-z-convergence = the hard ceiling on ANY recovery scheme. CMT bound γ_eff = γ_rad(1 − f·cos φ); the measured +0.002 pillar pair ⇒ f·cos φ ≈ 2 %. Gains scale **LINEARLY** in N phase-correct scatterers (not N²). Radial placement tolerance ±50–90 nm; λ-bandwidth a non-issue (±14–35 nm). Ranked: (1) short interface taper at the π-shift (Q_rad grows EXPONENTIALLY with taper periods, mode length only LINEARLY), (2) width increase to pull n_eff off the cladding light line, (3) k-space diagnostic first, (4) two in-line π-shifts subradiant supermode (d = m·0.544 µm, sweet spot ≈43 periods — but the composite mode spans ~22 µm, violating the width spec), (5) coherent chirped Bragg TRENCH arcs, (6) tooth shape at fixed κ. DEAD ENDS: far-field redirection, mode-gap weak modulation, bottom multilayer mirrors. **Multipole cancellation** (Johnson/Fan APL 78,3388; Nakamura/Asano/Noda OE 24,9541) is the real k-space version: a cancellation-capable perturbation must be EVEN about the defect AND contain sign alternation of Δε·E, i.e. paired features displaced ~half-period (≈258 nm) | — | reference_loss_reduction_options |

### 2.8 Convergence and numerics

| study | what was tested | verdict | key numbers | source |
|---|---|---|---|---|
| Transverse domain size | y/z span multiplier 1.8λ vs 2.7λ | **1.8λ kept for both polarizations** | TM is weakly confined (|E|² peaks just outside the core, tail ~20 dB/µm) → at 1.8λ it reaches the PML at only **−27 dB** (TE −39 dB); the −40 dB guideline needs M ≈ 2.66. But observables barely move: TM 1.8→2.7 changed loss_res 0.0409→0.0384 (~6 %), λ_res +0.05 nm, **Q ~767 unchanged**. Cost ≈ 2× runtime; 4.0 OOMs the default host RAM. **M ≤ 1.0λ corrupts the ports (T>1, negative loss); 1.5λ is the practical floor.** RAM is driven by grid AND port monitors (cross-section × n_wl_points): the same 5λ far-field run was **162 GB at 6001 pts / 150 nm** but **58 GB at 2001 pts / 10 nm** | project_transverse_domain_size_decision |
| EXCEPTION 1 — wide / near-cutoff modes | 80 µm mode on a 200 nm TM guide | **1.8 is NOT safe there** | n_eff 1.4585, only +0.0145 above clad → tail decays ~1.2 µm, mode hits the PML at **−10 dB even at 1.8** → unphysical peak **T 1.1–1.25 with NEGATIVE loss**. SPAN_MULT ≈ 4 recovers physical T ~0.78 (−22 dB); 5 gives −28 dB. The spatial FWHM is self-normalized and stays VALID despite T>1 | project_transverse_domain_size_decision, project_tm_wide_mode_corr |
| EXCEPTION 2 — TM corr-400 + far field | the 5.0λ far-field default | **not sufficient — use the converged box** | `y_span_override = 6.8 µm` + `span_multiplier_override = 5.42` (z ≈ 8.8 µm). Small box gave T 0.799 / loss 0.19 (z-PML artifact); converged truth **T 0.886 / loss 0.110** (jobs **116854/116870**; reproduced by scat_a_baseline 0.8862). Far-field monitor x-span default 30 µm CLIPS the corr-400 grazing lobe (peak at ux = 0.99) — the scatterers program uses 60 µm | project_transverse_domain_size_decision |
| corr-325 transverse-box convergence (7 tasks) | y-ladder 4.8/5.8/6.8 at z=8.8; z-ladder 5.8/6.8/10.8 at y=6.8; corner 5.8/6.8 | **CLOSED — campaign box = y 6.8 / z 6.8 µm (46.2 µm², −34 % cells vs the inherited 70.4)** | (y,z) → T / Q / Q_i / mode µm: (6.8,10.8) 0.9102/1762/38348/19.24 · (6.8,8.8) 0.9104/1760/38398/19.24 · (5.8,8.8) 0.9116/1761/38963/19.24 · (4.8,8.8) 0.9194/1767/42957/19.24 · (6.8,6.8) 0.9102/1760/38306/19.25 · (5.8,6.8) 0.9114/1760/38816/19.25 · (6.8,5.8) 0.9091/1758/37793/19.25. Stored ref (8.0,8.8) = 0.910/1760/19.24. **z CONVERGES at 6.8** (the z requirement is RADIATION-driven, not an evanescent-tail property); z=5.8 is the first bad rung. **y needs 6.8**: y=4.8 is a textbook grazing-PML artifact (T rises +0.0090 while the radiated fraction FALLS 0.0871→0.0785 — the near-axial lobe reflects off the y-PML and re-couples, faking +12 % Q_i). y=5.8 REJECTED despite passing the T floor (Q_i biased +1.1 %). **RULE: judge box convergence on Q_i, not T** — Q_i is ~**11.4×** more sensitive to a T error than T is ((1/2√T)/(1−√T) at T 0.91); Q_loaded is useless (1758–1767 across every box). Axes proven SEPARABLE. Open flanks accepted: decorations never tested (bare-device box is a LOWER bound for a decorated device); y>8.0 never tested at corr-325; opt mesh only; more PML LAYERS untested as a cheaper margin | Athena **131348**, `results_from_athena/tm_span_conv_c325/` | project_tm_nladder_surrogate |
| TM mesh convergence | `CONV_POL=TM`, Phase X cells [4..9] at dyz=50, Phase YZ dyz [50,25,10], metric Q (2 % threshold), calibrated pitch 518.3 / n 1.9963 / window 1567±20 nm | **dz is the load-bearing axis for TM; dz=25 is the conservative shared choice** | Phase X (dyz=50): cells 4/5/6 → Q 766.7/820.9/832.4, λ 1573.9/1571.4/1569.7. TM Q drifts ~1 %/halving (50→25 +1.06 %, 25→10 +1.02 %) → TM rule picks dz=25; TE Q is <1 % converged at dz=50. λ-convergence ~same for both (cells 5–7). TM accuracy limited by **dz** (E_z discontinuity at horizontal core interfaces), TE by **dx** (sidewall corrugation). **TE convergence data is NOT a checkpoint** — it lives in `C:\Users\evyat\OneDrive\Documents\תואר שני\Photonics Research\simulation_stats\mesh_convergence_results.xlsx` (sheet "Mesh Convergence"), older divisor-based dz, 1.977/1.44, Λ=500: cells 4/5/6/7/8/10 → Q 1322/1404/1453/1490/1521/1564, λ 1563.5→1555.9 nm | job **94750** (node n318, RTX PRO 6000 Blackwell), `/work/results/mesh_convergence_tm/` | project_tm_convergence_study |
| Auto-shutoff threshold | `autoshutoff_qspan`: 3 TM plain/trench devices Q 2.5k–26.7k × {1e-4..1e-7} + TE N80 + apod-20 × {1e-5..1e-7} | **SETTLED — production 1e-7 for EVERYTHING** | **truncation error is a function of Q ONLY** — five devices, two polarizations, three families collapse on one curve. At shutoff 1e-6: dQ/Q = −1.7 % (Q 1.28k) → −2.1 % (1.4k) → −3.3 % (2.5k) → −9.9 % (13.9k) → **−15.7 % (26.7k)**; scales ~Q^0.7; T errors −0.014..−0.030. 1e-6 passes the 2 %Q / 0.015T gate only for Q ≲ 1.5k. **1e-8 is UNREACHABLE** — the measured physical energy floor is a ~5e-8 total-field plateau, so the criterion never fires (3 tasks cancelled at the 2000 ps cap). Fine print: 1e-7 itself may sit ~3 % below infinite-time Q (DERIVED) | IGUM **49537** + **49718** (16 grid points + 2 anchors); KEEP-FOREVER data `results_from_igum/autoshutoff_qspan/results/` | project_autoshutoff_verdict |
| Si handle-wafer / fab-stack check | Si / 3.8 µm BOX / SiN / oxide vs our all-oxide z-symmetric model; plain W800 ± Si and full-height trench ± Si | **CLOSED DOUBLE NULL — the Si handle wafer is invisible at our floor; feature DROPPED** | ctrl (noZsym) T 0.8862 / λ 1558.616 · Si BOX3800 0.8860 (dT −0.0001) · Si BOX3665 0.8855 (−0.0007) · trench full-z 0.9037 / 1557.846 (+0.0175) · trench+Si 0.9039 (+0.0002 vs trench). λ pinned to 1 pm. Row 5 (trench to 3.8 µm with OXIDE below): T 0.9035 ⇒ **three-way tie** full-z 0.9037 / on-Si 0.9039 / oxide-floor 0.9035 (spread 0.0004 = ¼ floor) ⇒ trench depth beyond ~3.8 µm is irrelevant in ANY stack. Bonus: z-sym-OFF + IGUM + box8 reproduced the Athena z-sym-ON box6.8 benchmark EXACTLY (dT −0.0000, dλ +0.005 nm). z-PML ladder (existing data): oxide-below-PML 1.43/1.93/2.73/3.23/4.23/5.23 µm → T 0.7926/0.8671/0.8833/0.8851/0.8860/0.8860, plateau from 4.2 µm. **FINAL OP RULE: keep the symmetric all-oxide model at the standard box for everything, incl. full-z trenches.** Feature reverted (commits 22e5e68 + fbad9ad); restore recipe and the three load-bearing pieces (port-window clip to Si_top+0.2 µm, z-symmetry OFF, trench z-clip at the Si face) are in the memory file | IGUM **43459** + **43519** (5/5) + Athena **126104_5**; `results_from_igum/si_substrate_check/results/` | project_si_substrate_check |
| ★caveat on the z-symmetry null | — | corr/N-SPECIFIC, not general | see §2.1 flush-trench row: at corr 325 / N165 the same z-sym-OFF change moved λ +2.10 nm, T −0.21 dB, Q +7 % | project_trench_flush_top_study |

### 2.9 Side-by-side coupled cavities

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Geometry / machinery | `n_devices=2`: device 1 driven at y=+s/2, x=0; device 2 passive plain grating at y=−s/2, x=Δx; s = `device_gap_m` + ½W_wide1 + ½W_wide2 (gap = wide-tooth edge-to-edge). y-symmetry FORCED OFF, z kept. 4 ports: Port_1 (S11=R), Port_2 (S21=T), Port_3/4 = raw |S31|²/|S41|² into device 2 (no phase correction); `loss_4port = 1−R−T−C3−C4` | — | single-device path verified bit-identical | `runners/side_by_side/` | project_side_by_side_coupling |
| TE/TM headline grid (48 tasks) | gap {1000,1500,2000} nm × stagger {0,500,1000,2000,3000,4000,6000,8000} nm × {TE,TM}, both devices corr 500, pitch 500, N=80, n_core 1.977 | **CLOSED — TE and TM couple by DIFFERENT mechanisms** | **TE = evanescent** (directional-coupler): coupling 0.59 → 0.19 → 0.019 at gap 1.0/1.5/2.0 µm (≈30× drop, exponential), DECREASES with stagger, ~gone by 2 µm ⇒ TE side-by-side radiative coupling is negligible (matches the theory sideways null). **TM = genuine radiative tail**: weak gap decay 0.37 → 0.30 → 0.24 (≈1.5×), and at 1 µm gap coupling RISES with stagger 0.37 → 0.42 (device 2 slides into device 1's in-plane forward lobe) — vindicates Paper 8's directional argument, effect modest, no sharp λ₀/2n_c periodicity at coarse Δx. Q: TE ~440–690, TM ~2440–2900 (no dramatic subradiant boost — that needs the IN-LINE pair). Practical: use gap ≥1.5–2 µm to kill TE evanescent coupling | job **110724** (48/48) after failures **110253/110259**; `results_from_athena/side_by_side_coupling/FINDINGS.md` + maps_TE/TM.png | project_side_by_side_coupling |
| TM-400 gap × stagger (42 tasks) | current TM device (pitch 516.83, corr 400, h350) gaps {1000..2000} × Δx {0..6000}, window 1558.5 / 30 nm / 3001 | done | device-1 peak T rises with gap + stagger, MAX **0.611 at gap 2.0 µm / Δx 6 µm** (still < the isolated single-device 0.827 ⇒ TM does NOT fully decouple by 2 µm, unlike TE). Supermode splitting clean and gap-driven: **7.6 nm (gap 1.0) → 1.4 nm (gap 2.0)** at Δx=0, decaying ×0.71 per +200 nm. **TM splits MORE than TE at the same gap** (TE 4.5 nm @1.0 µm, single-peak by 1.8 µm) = TM is the longer-range, radiatively-coupling polarization. Q ~1100–1500, outliers to ~3000 | job **114387** (42/42), `results_from_athena/side_by_side_tm_400nm/`, scripts `plot_tm_400nm_map.py` / `plot_tm_400nm_splitting.py` | project_side_by_side_coupling |
| Wide-gap extension (35 tasks) | gaps {2200,2400,2600,2800,3000} × same 7 staggers, TM | **TM DECOUPLES at gap ≥ 2.4 µm** | splitting Δλ(Δx=0): 7.6 → 1.4 (2.0) → 0.8 (2.2) → **0 from 2.4 µm** (single peak, 0/7 two-peak at ≥2.4) vs TE single-peak by ~1.8 µm ⇒ TM coupling reaches ~0.6 µm further. Peak T recovers to **0.74 at gap 2.8–3.0 µm / Δx 6 µm** (~90 % of the isolated 0.827). KEY SUBTLETY: splitting closes (2.4 µm) BEFORE T saturates — residual coupling/loss keeps T under 0.827 even with no splitting; T would need ~3.5–4 µm | job **114498** (35/35); the 35 .mat were COPIED into `side_by_side_tm_400nm/results` (77 files, gaps 1.0–3.0 µm) | project_side_by_side_coupling |
| detune (Friedrich–Wintgen) sibling | `corrugation_depth_2_nm` {400..600} at fixed (gap, Δx) × {TE,TM} = 14 tasks; and a CLOSED device-2 recycler grid (48 tasks) | **NOT launched / deferred** | detune runner pre-set to gap 1000 / Δx 8000 (the TM coupling max) | — | project_side_by_side_coupling |
| ★incident that produced a standing rule | — | — | first-run failures **110253/110259**: every config built the SAME `layout_N80_avg.fsp` / `_output.h5`, so concurrent array tasks on one node clobbered each other mid-run → empty result → `getresult("…Port_1","expansion for port monitor")` LumApiError. FIX = unique per-config filename via `generate_file_tag`. Second: deploying the detune job right after the grid overwrote the shared `data/sweep_list.txt` (48 → 14 lines) → "SWEEP_INDEX out of range" on the grid's later tasks | — | project_side_by_side_coupling |

### 2.10 Grating-coupler sibling project (separate repo)

| item | content | source |
|---|---|---|
| where | **`C:\Users\evyat\Lumerical\grating_coupler_FDTD_codes`** (started 2026-05-18) — TM-polarized grating coupler on oxide-clad LPCVD Si3N4 (350 nm core, 3.8 µm SiO2 BOX, 4 µm Si substrate, oxide top cladding as FDTD background). It is NOT the pi-shift Bragg device; it reuses this repo's `simulation_config.py` dataclass pattern, `athena/deploy_athena.sh`, and the `runners/<X>/` `BASE`+`SPEC` contract | project_tm_grating_coupler_sibling |
| locked decisions | optimizer stack = Lumerical built-in PSO (`addsweep` type=Optimization) for the coarse global seed → lumopt L-BFGS-B adjoint refinement; background = `SiO2 (Glass) - Palik`, Si substrate the only explicit object below the BOX, no air anywhere; material `Si3N4 (Silicon Nitride) - Luke`; GDS regenerated radial/focused in nazca from the optimized (Λᵢ, wᵢ) arrays (FDTD optimization is straight-tooth). **TM is the priority deliverable, TE inverse design is stretch** | project_tm_grating_coupler_sibling |
| dual deliverable (do not drop either) | **1) uniform grating** — one (pitch, fill_factor, fiber_angle) from native PSO, ~3 parameters, trivially sweepable for fab-tolerance / pitch-sensitivity / angle-acceptance tables → `results/tm_grating_coupler_uniform.gds`; **2) inverse-design grating** — 40 per-tooth (a_i, b_i) from lumopt adjoint, higher peak but opaque for design metrics → `results/tm_grating_coupler_inverse.gds`. Plus a comparison summary (peak coupling, 1 dB BW, pitch tolerance, angle tolerance). User 2026-05-20: "once you have a uniform grading, then it's easy to think of the main design metrics" | project_gc_dual_deliverable |
| 2D adjoint | **OPTIONAL, not required** — the Ansys 3D KB recipe seeds `ParameterizedGeometry` directly from a uniform analytical/PSO start; the 2D stage costs 3–5 h of Athena CPU (Lumerical GPU is 3D-only) and competes with real 3D iterations. For ≤8 h windows: forward verification → PSO uniform (cheap 2D CPU) → 3D adjoint seeded from PSO | project_gc_skip_2d_adjoint |
| ★BUG FIXED — source phi | `gc_device._add_fiber_source` had `fdtd.set("angle phi", -90)` since commit 7ef3342. With injection axis=y / direction=Backward, φ=−90° tilts toward −z, **perpendicular to the waveguide**. For a 1D grating with teeth in z and propagation in x the tilt must be in the xy plane ⇒ **φ = 0** (with θ = −fiber_angle_deg). This explains why "successful" forward sims (job **82066**) peaked at −40 dB and why TM at 37° aborted the FDTD engine entirely (job **82283**). Fixed 2026-05-20; this cost weeks | project_gc_source_phi_bug |

### 2.11 The Itai Lev-Ran collaboration

| item | content | source |
|---|---|---|
| IT15 `custom_params_HighBulk_HighTrans_ADW` ("HH") | corrugation **500 nm**, **98 periods**, pitch **514 nm**, avg wg **1.0 µm**, **N_t = 60** apodized, `dw_middle = 0`, `apod_method="custom"`, target FWHM 15.1 µm, target λ **1600 nm**, `delta_pitch_middle = [0,−10,−20,−30]` (we only ever ran dpm=0), advanced index + dw correction ON. Full device length 101 µm; apodized extent 31 µm/side | reference_itai_it15_designs |
| IT15 `A8T500_advanced_index_corrected_params_ADW` (tanh sibling) | corrugation 500 nm, 100 periods, pitch 528 nm, avg wg 0.8 µm, N_t = 20, `dw_middle = 4 nm`, tanh a = 0.4, target FWHM 22 µm. DERIVED: 18 of the 20 apodized teeth are below 90 % of full corrugation; depths 4 / 122 / 262 / 383 / 483 nm at teeth 1/5/10/15/20 | reference_itai_it15_designs |
| ★"NEVER RECEIVED" — SUPERSEDED | the earlier memory said `Nt60_highbulk_HighCorr_dw.npy`, `Part_Itai.py`, `helpers.py` were absent. **All source files ARE on disk** at `C:\Users\evyat\OneDrive\Documents\תואר שני\Photonics Research\Results for lab\share_with_Evyatar\` | project_itai_hh_apodization (supersedes reference_itai_it15_designs) |
| The profile is **NOT a taper** | 61 Δw values inward: 0 at the cavity → **overshoot 1200 nm (2.4× the 500 nm bulk) at d≈24–26** → dip 603 → second lobe 1188 at d≈50–54 → bulk from d=61. Drawn: bulk 746.9/1257.1 nm, extremes 577/1915 nm, cavity pitch/2 at 950.3 nm, avg 1.0 µm, pitch 514, 98 periods/side, **no tooth shift**. Only AMPLITUDE moves the mode width; **N sets peak T** | project_itai_hh_apodization, reference_itai_npy_analysis_recipe |
| ★THE DEFECT THAT VOIDED ROUND 1 | round 1 returned **T+R up to 1.045** at resonance. Cause: the transverse domain was too small and **z was the culprit, not y** — growing y 3.99→4.44 µm made it WORSE (T 1.0255→1.0426); growing z 3.16→5.03 µm restored T+R = 0.9576. Aggravated by a real code defect: `simulation_config.py:530` sizes y_span from the SCALAR `width_wide_m` (avg+corr/2 = 1184 nm) and never consults the per-tooth arrays whose real max is 1632 nm — 448 nm of intended standoff eaten silently; same blind spot at `bragg_device.py:821`. Both fixed locally, snapshot gate byte-identical on all 6 configs. The inverse-design programme was NEVER exposed (it sets `y_span_override_m`, box 6.8 × 6.81). **Jobs 63237 / 63424 (round 1) are VOID** | project_itai_hh_apodization |
| TE results (box 6.8×6.81 unless noted, all T+R<1) | scale 0.72 @ 6.0×5.0: mode 17.614, Q_L 10618, T 0.9536, T+R 0.9576, Q_i 452475 · 0.72 @ 7.5×6.9: 17.615 / 10662 / 0.9566 / 0.9582 / 486033 · 0.72 @ 9.0×8.8: 17.612 / 10667 / 0.9541 / 0.9585 / 458925 · scale 0.52: 20.720 / 2236 / 0.98821 / 0.98836 / 378044 · scale 0.58: 19.688 / 3452 / 0.98393 / 0.98432 / 427874. Five measurements cluster at Q_i 378k–486k (±12 %). Interpolated to exactly 20.0 µm: scale ≈ 0.562, Q_i ≈ 413000 → Q(−3 dB) ≈ 121000 vs our 12903 ≈ **9.4×**. Final TE ladder (scale 0.58): N = 98/140/155/175/195 → Q_L 3452/11449/17117/29247/49166, T 0.984/0.963/0.941/0.909/0.860, Q_i 427874/621270/570920/631194/674598, mode 19.69–19.86 µm; Q_c growth 2.85 %/period, T=0.5 crossing extrapolates to N=254; best-conditioned row (N=195, 7.4 % error) ⇒ **Q_i 674598 → Q(−3 dB) 197657 = 15.5× our 12741** (treat as a FLOOR, trend still upward) | project_itai_hh_apodization |
| TM results (crossing ladder, well conditioned, Q_i N-INDEPENDENT on HIS geometry) | N 98 → mode 20.099, Q_L 2614, T 0.95058, T+R 0.95125, Q_i 104458 · N 110 → 20.184, 3914, Q_c 4057, 0.93074, 0.93204, 111029 · N 124 → 20.250, 5991, 6312, 0.90075, 0.90339, 117649 · N 140 → 20.297, 9580, 10404, 0.84792, 0.85423, 121007. Q_c grows 3.19 %/period; T=0.5 crossing at N=189 → Q(−3 dB) = 34153 vs our 13930 = 2.45×. **Measured directly at N=189: T 0.5275, Q_L 34006; at exactly T=0.5 → Q 36400 = 2.6× our 13930** (direct-crossing and Q_i routes agree to 0.4 %). As-drawn TM (scale 1.0, pitch 514, box 9.0×8.8): λ 1564.664, Q_L 8038, T 0.7205, T+R 0.7427, Q_i 53174, mode 17.222 | project_itai_hh_apodization |
| His FAB device (user-supplied) | Q_L 77000, peak 5 dB below top → T = 0.316, Q_i = 175936 at ~15 µm mode. The operating point CANCELS in the ratio (Q_L = Q_i(1−√T)), so his −5 dB vs our −3 dB is irrelevant: **4.05× our device at any T**, and 7.9–10.7× once our device is corrected to his 15.27 µm. **OUR SIM OF HIS DEVICE READS 2.9× LOW vs his fab** (Q_i 60965 vs 175936) — suspected dx=50 nm staircasing of sidewalls swinging 577→1929 nm. UNTESTED | project_itai_hh_apodization |
| Jobs (IGUM) | **63237 / 63424 round 1 = VOID** (undersized box, T+R>1) · **63438** box ladder (TE rungs 4.4/6.0/7.5/9.0 DONE/converged; TM rungs + our TE baseline died on a license cascade) · **63441** as-drawn · **63451** TM crossing ladder N=110/124/140 · **63454** round 2 · **63491** task 3 = TM N=189 direct crossing · **63722** = his re-optimized Nt60_FWHM_20 at scale 1.0. Runners `runners/sweeps/{itai_hh_apod,itai_hh_asdrawn,itai_hh_tm_cross}.py`; reduced extractor on IGUM at `/tmp/hh_extract.py` (never pull the ~700 MB volumes) | project_itai_hh_apodization |
| ★THE RULER (standing recipe) | Every delivery = a raw Δw `.npy` (61 values, nm) + a `config_params` snippet. (1) **Array ordering**: stored bulk-first (index 0 = outermost transition ≈ bulk Δw; index 60 = cavity = 0.0); our runners are cavity-first ⇒ `profile[::-1]`; check `npy[-1]==0` and `npy[0]≈corr_depth`. (2) **His drawing chain** (`apod_method='custom'`): fill, then `advanced_dw_correction` (3-D LUT `dw_correction_LUT.mat`, maps 500 → **510.2 nm** at λ1600/w0 1.0), then `advanced_index_correction` (`neff_LUT_2D.mat`, fsolve the per-period MID width so the period-averaged n_eff equals bulk), then `helpers.round_gds` on a 0.1 nm grid. (3) **THE GATE, never skip**: run the chain on his OLD .npy and compare to `runners/sweeps/itai_hh_apod.py`'s `APOD_NARROW_NM`/`APOD_WIDE_NM` — **Δw must match to ≤0.1 nm**; a constant **−2.0 nm common-mode offset on the MID width is EXPECTED** (≈0.3 nm of λ, absorbed by the pitch retune), a Δw mismatch is not. (4) delivered at scale 1.0 ⇒ use HIS drawn widths; if WE scale, his index correction breaks and the mid width must be re-solved with OUR FDE curve. (5) **Pitch retune**: envelope-weighted ⟨n⟩ (`pitch_for`) — per-period (narrow+wide)/2 n_eff averaged over a 20 µm Gaussian on the cavity, `pitch = λ/(K·2⟨n⟩)`; MEASURED accuracy **−0.16 nm (TE), −0.42 nm (TM)**; N-independent. Worked: old ×0.562 → ~500 nm; new Nt60_FWHM_20 → **491.06 nm**. (6) **HIS MODEL IS ACCURATE AT SCALE 1.0**: IGUM **63722** predicted 19.9889 µm, FDTD measured **19.633 µm = −1.8 %**. The old ~13 % gap priced OUR scaling, not his model. Our scale ladder: 0.78 → 16.88 µm (his model said ~20), 0.72 → 17.61, 0.58 → 19.688, 0.52 → 20.720; slope **−1.98 µm per 0.1 of scale** in the 20 µm region. (7) Non-negotiable numerics: set the box explicitly (**y 6.8 µm / span_mult 4.14 → z 6.81 µm**), always gate on T+R<1, measure Q_i near T 0.5–0.8, sweep path fixes `n_wl_points = 3001` | reference_itai_npy_analysis_recipe |
| method facts worth keeping | `fwhm_m` is **box-INDEPENDENT** (17.61 µm at every box, to 0.01); λ_res likewise robust. `Q_i = Q_L/(1−√T)` — at T=0.95 a 1 % T error is a **20 %** Q_i error, at T=0.5 only 2.4 %. **Q_i is N-independent** (our own ladder 43205/43257/42994/41716 across N 166–215 while Q_L varies 2.3×) ⇒ **Q(−3 dB) = 0.293·Q_i is a MEASURED relation**. The **TE crossing method is impractical** (N≈177, Q_L 132k, 11.8 pm linewidth, 1760 ps ring-down ⇒ ~40 h/point); TM crossing is fine (N≈124, 207 ps, ~3.4 h). **The 1D TMM misled the scale ladder** (predicted 0.78→~20 µm, reality 0.72→17.61) — measure one row and use the local slope | project_itai_hh_apodization |
| ★incident | the array throttle was raised to 4 at **37/50 seats** (rule says hold at ≥35) → 4 tasks died with the bare `in run:` IGUM license-starvation signature; cost more than the fan-out saved | project_itai_hh_apodization |

### 2.12 Far field

| study | what was tested | verdict | key numbers | jobs / dir | source |
|---|---|---|---|---|---|
| Far-field spherical-harmonic (multipole) study at 20 µm / N=98 | three devices at the SAME mode width and length: (A) plain TE corr 250 / pitch 500 / W800, (B) Itai's re-optimized Nt60 overshoot apodization untouched (`runners/sweeps/itai_hh_nt60w20.teeth(98)`, pitch 491.06), (C) plain TM corr 325 / pitch 516.83 / W800; complex far field at resonance → power fraction per vector spherical harmonic (E/M, l, m) | **ROUND A COMPLETE 2026-09-29 — see Part 5 §5.1 for the measured outcome** (the text below is the pre-dispatch registration) | Round A = 6 rows: TE far-field BOX LADDER on A at y/z 6.8/6.81, 8.0/8.8, 10.0/10.8, 12.0/12.8 (first-ever TE far-field box convergence) + B at 6.8/6.81 (identical numerics to 63722 ⇒ that row is its control) + C at 8.0/8.8. Windows on stored resonances: **A 1559.986** (MEASURED at box 6.8, `result_N166_avg_C250_Ybox6p8_Zbox6p8.mat` — +0.20 nm vs the default-box 1559.79), **B 1559.8597**, **C 1559.006**. FF monitors x-span 80 µm, 401² grid, complex, 81 freq points, SBATCH_MEM=200G, `--max-concurrent=3`. Mode widths EXPECTED at N=98: A ~20.0, B 19.63, C ~19.2 µm. Verdict rule: last two rungs agree (multipole spectrum, T, T+R, E²-weighted mean \|ux\|) within rung jitter ⇒ box chosen | Athena **164883** smoke FAIL (`apply_monitor_overrides` in `sim_helpers` reset the FF monitors to 1 point — FIXED; 2D field planes were 197 MB at N=10 ⇒ `record_2d_fields` OFF), **164891** smoke PASS, **164893** = round A, 6 tasks, %3, 200 G, dispatched ~20:40. Results → `results/farfield_sph_20um/results/` on Athena | project_farfield_sph_20um |
| ★engine change that came with it | `FarFieldConfig.farfield_freq_points` (default 1 = legacy band-centre; >1 = record the band and project at the recorded point nearest `resonance_wavelength_nm`) | default-inert, snapshot gate 6/6 identical | **every stored `*_ff.mat` was projected at the BAND CENTRE** — one stored TE example was 41 % of a linewidth off resonance | — | project_farfield_sph_20um |
| tool | `python_tools/farfield_multipole.py <mat> [--lmax 200] [--csv]` — full sphere from top (+z) + side (+y) monitors, nearest-normal patchwork, parities READ from the data, Jackson X_lm / n×X_lm projection | VALIDATED | Legendre + derivative vs scipy; five synthetic dipoles → 100 % in l=1 with correct E/M type; Parseval 1.000; parities correct. On a real stored TE far field (scat_z_teffmap N80 corr300): transversality 2e-6, Parseval 0.998, s_y=−1 s_z=+1, **l ≤ 5 carries 88 %** (power-weighted ⟨l⟩ 5.5) — the pattern is LOW order, not kR~100 as feared | — | project_farfield_sph_20um |

### 2.13 TM length ladders and wide-mode studies

| study | what was tested | verdict | key numbers | source |
|---|---|---|---|---|
| TM period-match to TE@80 | integer bisection on TM N until peak T equals TE@80's | **CLOSED** | TE@80 T 0.8304 / FWHM 0.917 nm / Q ≈ 1712 / λ 1571.00; matched **TM N = 132/side** (crossing 131.6): T 0.8285, FWHM 0.320 nm, **Q ≈ 4910**, λ 1570.75 ⇒ **~2.9× TE's Q at equal peak T**. TM@80 Q ≈ 803. Physics: lossless real indices ⇒ peak T<1 is RADIATION loss; TM couples weakly so it needs 132 vs 80 periods for the same grating strength, and that longer weak grating gives a much narrower resonance. GOTCHA: `spectral_fwhm_nm` is stored **NEGATIVE** — use Q = λ/\|spectral_fwhm_nm\| | project_tm_period_match_te |
| TM corrugation ↔ mode-width match | bisect TM corrugation to match TE's spatial mode width (= same κ) at N=80 | **CLOSED — match = 400.0 nm** | TM FWHM(corr): 300 → 19.26 µm, 350 → 17.24, **400 → 15.549**, 450 → 13.86; target TE@300 = 15.544 µm ⇒ 400 nm lands Δ = **+0.004 µm (+0.03 %)**, interpolated crossing 400.1 nm. **Cost of matching width: TM peak T 0.93 → 0.83, Q 729 → 1267** (vs TE Q 1459). TM couples ~1/3 weaker and needs ~1.33× the depth. Physics: envelope \|E\| ∝ exp(−κ\|x\|) ⇒ energy FWHM = **ln2/κ**, depends ONLY on κ, not on device length (given κL ≫ 1) ⇒ same κ ⇔ same mode width; N·Λ sets κL → linewidth / Q / peak T, NOT the spatial width. Job **113571**; outputs `results_from_athena/tm_match_corr/`; NOT retrofitted into existing TM studies designed at corr 300 | project_tm_corrugation_match_modewidth |
| TM surrogate-N ladder | how short a device inverse design may use | **CLOSED — corr-325 surrogate N=100; equivalence rule = match 2κL, not N** | see §1.4 for the ladders. Binding constraints are NOT κL>1 (even N=60 has κL 1.09 and a clean resonance) but (1) **mode truncation** (84/89/92/96/98 % at N 60/70/80/100/120; intrinsic Q rises 24k→44k across the ladder as truncated tails radiate) and (2) **loss visibility** (1−T is only 3.3–4.8 % at N≤80, so a 10 %-relative loss gain moves T by 0.003–0.005 ≈ the 0.0018 jitter floor; at N=100 it is 0.009 = 5× floor, at N=120 0.0156 = 8.7×). N=120 is the **escalation rung**. **corr-400 verdict: N=80 was right all along** — it sits JUST above the floor (lever 6.3×); N=70 usable (98 %, 4.5×), N=60 marginal (94 %, 3.4 %). User chose N=100 over N=90 for signal quality ("keep at 100, I don't like the truncation"). Winners are always §2-confirmed at production N (165–169) + accurate mesh. Zero-GPU truncation model F(N) = F∞ − B·exp(−2κNΛ) VALIDATED: fit on N 60/70/80 predicted held-out N=100/120 to −0.08 / −0.13 µm (0.4–0.7 %, slight under-estimate — add ~+0.1 µm) | project_tm_nladder_surrogate |
| TM wide-mode corrugation search (h350 and h200) | secant search in the LINEAR coordinate 1/fwhm for a target spatial FWHM (default 80 µm) at N=300/side | **h200 anchored at two wavelengths** | h350 pitch 516.83 eval 1: corr 60 → **91.0 µm** at λ 1561.2, T 0.95, then paused. h200: 1550 nm target, pitch 531.5, SPAN_MULT=4, SBATCH_MEM=256G → **corr 227.7 nm → 80.18 µm** (Q ~1659, T 0.71, ~16 % radiation), job **114496**. Retarget to 1590 nm: phase A secant on pitch (job **114795**) → **pitch 545.959 nm** (λ_res 1589.93, Δ −0.07); phase B (job **114830**) → **corr 262.7 nm → 79.75 µm** (Δ −0.31 %), λ_res 1590.02, **Q 1554, T 0.562, R 0.032, loss 0.406** (T+R+loss = 1.000, all physical, max T 0.756 ≤ 1). Study dir `tm_wide_mode_H200_P546`. A wide mode ALSO needs a long device (80 µm FWHM is ±40 µm before its exp tails; half-device must be ≳2× the target FWHM; N=300/side ≈ 3.9 FWHM ≈ 93 % contained). Earlier runs 114361/114371 (crash: empty-env `float("")`), 114385/114399 (SM1.8, hit the corr cap at 83.5 µm + T>1) | project_tm_wide_mode_corr |
| ★operational traps from the wide-mode work | — | keep | `STUDY_DIR_NAME` must be HEIGHT- and PITCH-aware (`tm_wide_mode_H{h}_P{round(pitch)}`) because the eval cache stamps files `result_corrmatch_tm_C<pm>.mat` by **corrugation only**. Pitch must come via `TM_WIDE_PITCH_NM` (NOT `TM_PITCH_NM`, which deploy forwards as 500). **Never pass comma-containing values through `sbatch --export`** — `TM_WIDE_SEEDS_NM="227.7,250"` was truncated to `(227.7,)` and crashed the secant with `IndexError` AFTER a completed 23-minute GPU eval (now the 2nd point is synthesized from `corr2 = corr1·fwhm1/target`). Use `.get(key) or default`, never `.get(key, default)` — deploy exports empty strings and `float("")` crashes | project_tm_wide_mode_corr |
| h200 / w1800 / p504 / N=1300 study | exactly N=1300/side at accurate mesh, with convergence checks | **CLOSED — Q 7.0×10⁵, radiation-limited, resonance DARK; the wide-mode regime is off-spec anyway** | see §1.4 for the numbers. v1 array **127302** INVALID (N150 showed T 1.54–1.89, no stopband in 1520–1570: the port picks the true TM fundamental but the FDTD-mesh n_eff is 1.476 vs FDE 1.535 — a thin-core dz red-shift from the hardcoded `core_height/7` — plus ~2.5 % mode power on the PML at 1.8λ). v2 **127309** COMPLETE: λ 1492.122 identical at N=150/300 and span 3.8/5.0, stopband(T<0.75) 1488.4–1495.9, Q(N300) 1903/1883, span-5.0 physical T_res 0.9868, radiative loss 1.32 % ⇒ Q_rad ≈ 2.8e5 (DERIVED). Main run **127443** (a100-public, QOS **4d_1g**, 95 h limit, 200 G, SIM_TIME_PS=500, window 1489.1–1495.1) ran **71.5 h, exit 0**. Probe **127321** measured **610 wall-s per simulated ps** at the N1300 grid. Segfault in `getresult("field_profile")` (builder hardcoded 501 freq points ≈ 28 GB at 1.31 mm) → FIXED with the inert-by-default `cfg.monitors.field_profile_freq_points`. **Finder mis-pick AGAIN** (stored res 1490.162 = band edge, T = 1.024 > 1) — use the defect peak at 1492.124. At 20 ps the spectrum peaks at the upper BAND-EDGE lobe 1494.9 nm — do NOT read argmax as the defect at short T_sim. Datum: **max Q ever in ALL 1624 stored FDTD results = 17,745**. Open follow-up (never run): critical-coupling point N ≈ 500–700 → T ≈ 0.25–0.5 at Q ~3–4e5 | project_tm_h200_w1800_study |
| TM-vs-TE example / TM support | polarization plumbing | reference | `cfg.source.polarization` ("TE"/"TM") swaps port mode selection AND symmetry parity together — **TE: y-min Anti-Symmetric + z-min Symmetric; TM: y-min Symmetric + z-min Anti-Symmetric**. Three runners in `runners/tm/`: `run_te`, `run_tm`, `run_tm_vs_te`, all writing to ONE shared folder `results/tm_te/` via `STUDY_DIR_NAME = "tm_te"`. Far-field TE/TM pair: TE `tm_te/results/result_N80_avg_ff_te_fields_smp.mat` (1570.80 nm, pitch 500, Athena job **96422**, 25 min, 55 GB peak RAM); TM `run_tm/results/result_N80_TM_avg_ff_tm_P518p3_fields_smp.mat` (1570.60 nm, pitch 518.3). Literature expectation (verified vs Chen et al. OE 23, 25295): TM couples weakly to SIDEWALL corrugation (field at top/bottom interfaces) → much narrower TM stopband; "no stopband" is the literature-expected failure mode and the runner raises explicitly | project_tm_vs_te_example |

## 3. The device scoreboard — best known devices

All numbers below are MEASURED (read from the named stored `.mat` / study dir) unless labeled DERIVED or EXPECTED. Absolute T is numerics-sensitive: compare only within one mesher, one box, one window (CLAUDE.md §2).

## 3.1 The −3 dB / ~20 µm TM family (the production operating point)

Shared anchored geometry unless stated: **h350, pitch 516.83 nm, corrugation 325 nm, W800 avg cavity, n_core 1.97 / n_clad 1.444, box y8.0/z8.8 µm, window 20 nm @ 4001 pts, dx = 50 nm optimization mesh, conformal variant 0, auto-shutoff 1e-7, engine 2026 R1.3 build 4572.** "N" = `n_periods_each_side`.

| device | N | T (dB) | λ_res (nm) | Q_L | Q_i | mode width µm | jobs | stored data |
|---|---|---|---|---|---|---|---|---|
| **full-z air trench** W800/d1800/H12000, L 178 µm | 170 | 0.5021 (−2.99) | 1558.27 | **18777** (+35%) | 64.2k DERIVED @N165 | 19.57 | IGUM 47910/48458/48711/48973 | `results_from_igum/trench_q3db_20um/results/` |
| **comb PAIR** r110/d1.8/Λ531: 31 posts @ +12 µm (δx 88.5 nm) + 31 @ −12 µm (δx 177 nm), mirrored ±y, h350 | 172 | 0.5003 (−3.01) | — | **18093** | — | 19.76 | Athena 146639 → **146681** | `results_from_athena/comb_q3db_lock/` |
| single **61-post comb** r110/d1.8/Λ531/270° (ends ±16 µm) | 171 | 0.5060 (−2.96) | — | 17557 | — | 19.79 | Athena 146639 → 146681 | `results_from_athena/comb_q3db_lock/` |
| **flush-top trench** (air, w800/d1800, z −3.975→+0.175 µm, z-sym OFF + forced symmetric z mesh) | 168 | 0.5017 (−2.996) | 1558.482 | 16942 (+21.6%) | — | 19.68 | Athena 129105_1 (bracket N169 T 0.4912 / Q 17392, Athena 129730_2) | `results_from_athena/trench_flush_q3db/` |
| **comb lock (earlier)** Λ531/δx401 (270°)/r80/d1.9/57 posts/h350 | 169 | 0.4961 (−3.04) | 1559.011 | 16203 (+16.3%) | 55k (cited) | 19.91 | Athena 130458 + **130548** | `results_from_athena/comb_q3db/` (FINDINGS.md + benchmark figure) |
| **bare control** (no decoration) | 165 | 0.4906–0.491 (−3.09) | 1559.00 / 1559.001 | 13930 | 46.5k DERIVED | 19.97 | IGUM (trench study) ≡ Athena 130458_0 **exact cross-cluster repro** | `results_from_igum/trench_q3db_20um/results/` |
| **TE Q3dB** corr 250, pitch **500**, W800, h350, no trench | 166 | 0.4919 (−3.08) | 1559.79 | 12903 | 43205/43257/42994/41716 over N=166–215 (flat) | 20.46 | Athena 128580/128581/128593/128730/128733 | `results_from_athena/te_q3db_20um/results/result_N166_avg_C250.mat` |

**Current benchmark at −3 dB / 20 µm:** the **full-z air trench, Q_L 18777** — still the highest measured loaded Q at the operating point. The **2026-09-12 comb pair (Q 18093)** is statistically alongside it and is the best **single-litho, width-neutral** device (comb changes mode width by −0.35%; λ pull +10 pm). Pair vs single 61-post comb at −3 dB = +3.1% Q = one band width ⇒ **marginal, candidate only**; the 61-post single comb is the simpler device for the same result. TE and TM *bare* ceilings are near-identical (12903 vs 13930); the trench is a measured **TE null** (ctrl 0.8756 vs trench 0.8747).

Measured benchmark ordering as of 2026-08-11 (all identical numerics): `full-z trench 18777 (+35%) > flush 16942 (+21.6%) > comb 16203 (+16.3%) > ctrl 13930 > TE corr250 12903`, now joined by comb-pair 18093 and single-61 17557 at N=172/171.

### 3.2 The 14 µm device (off-spec width, delivered by prediction in 2 runs)

| row | corr nm | N | T (dB) | λ_res nm | Q_L | width µm | job |
|---|---|---|---|---|---|---|---|
| rung 0 (one-shot corr knob) | 448.4 | 98 | 0.5808 | 1557.75 | 3853 | 14.15 | IGUM **68086** |
| **anchored confirm** | 448.4 | **103** | **0.5097 (−3 dB)** | **1557.75** | **4644** | **14.19** | IGUM **68925** |

Stored: `results_from_igum/tm_q3db_14um_knob/results/`. Rung 0 missed the T band (0.4576–0.5280) but its **width knob worked to +2.1%** and Q_i was right to +3.8%; the whole error was in Q_c. One zero-GPU two-term fix + rung 0 as anchor gave the confirm to T −0.002 / Q_L −0.5%. Bonus MEASURED: the c448 pair's Q_c rate is 0.05040/period vs 0.05088 predicted by κ ∝ corr from c325 — **1.0% at +38% corrugation**.

### 3.3 Apodized TM (high Q_i, NOT at −3 dB — an open program decision)

Stage-M **apod-10, corr 400, N=150** control, re-read from files 2026-09-11: T 0.7624, R 0.0218, linewidth 56.5 pm, Q_L 27584, mode 20.29 µm ⇒ **Q_i ≈ 217k DERIVED** (~4× the comb lock's 55k). **+ full-z trench:** T 0.8944, Q_L 33861, mode 19.92 µm ⇒
**Q_i ≈ 620k DERIVED** — but only **9 points across the linewidth**, so that is a LOWER BOUND, not a measurement (see §4.8). Whether apodized TM enters the q3db benchmark is open (program item B3). Apodization does NOT stack with the comb (§3.5); the trench does (+0.0039 on apod-10).

### 3.4 The corr-400 / N=80 T-maximisation family (~15.5 µm modes, not the 20 µm spec)

Base: pitch 516.83, corr 400 (widths 600/1000), W800, h350, N=80/side, converged box y 6.8 µm / z 8.8 µm (span_mult 5.42). **Converged baseline MEASURED: T 0.8860, loss 0.1102, Q ≈ 1320, λ_res 1558.616** (jobs 116854 + 116870, keep-forever convergence data). Study controls in use: Athena 0.8851 / IGUM 0.8864 (both λ 1558.61, mode 15.53 µm); accurate-mesh ctrl 0.8787 at λ 1555.95. Jitter floor on T: **0.0018** at dx = 50 nm, 0.00004–0.0003 at accurate mesh.

| device | T | Δ vs own ctrl | loss | Q | λ_res nm | width µm | job | status |
|---|---|---|---|---|---|---|---|---|
| **W1050 cavity + comb pair** (+12/−12 µm, 134 nm) | **0.9374** | +0.0156 | — | — | — | — | Athena **146553** row 0 | CANDIDATE (opt mesh) |
| **narrow-touch** fused r=80 pairs at (x=0, y=±480) and (x=270, y=±380) | 0.9310 | +0.0448 | 0.0672 (−39%) | 1352 | 1558.90 | 16.02 (+3.2%); spectral FWHM 1.153 nm | Athena **121843** | CANDIDATE, single point, opt mesh |
| **comb pair Λ531** model phases (+12/60°, −12/120°) | 0.9070 | +0.0219 | — | — | — | — | Athena 146553 row 2 | CANDIDATE |
| **one centred 61-post comb** r110 Λ531 270° (ends ±16 µm) | 0.9064 | +0.0213 | — | — | — | — | Athena **146564** | CANDIDATE |
| comb pair Λ536 (+12/134 nm, −12/134 nm) | 0.9040 | +0.0189 | 0.093 | — | — | — | Athena **146419** row 0 | CANDIDATE |
| response-matrix pillar pair [0, 270] | 0.909 | +0.0227 | 0.0885 (−19.5%) | — | +70 pm | — | Athena 121239 | **historical only** — the 2-pillar pair is permanently dropped (CLAUDE.md §8) |
| **comb Λ531/270°(δx398)/r110/d1.8/h350, 31 posts** | 0.8966 | +0.0115 | — | ~1336 | — | 15.49 (−0.3%) | Athena **130154** (stage T) | **CONFIRMED** — §2 two-step passed: accurate mesh 0.8895 vs acc ctrl 0.8787 = +0.0108, mesh-twin Δ 0.0001 |
| comb N=41 posts, r=96 | 0.8999 | +0.0148 | — | — | — | — | Athena 130276 | CANDIDATE (two-step not run) |
| comb N=47 posts, r=89 | 0.9001 | +0.0150 | — | ~1338 | — | — | Athena 130397 | CANDIDATE |
| flush comb r=70 (top +0.175 / bottom −3.975, z-sym OFF, own zasym ctrl 0.8864) | 0.8990 | +0.0126 | — | — | — | — | Athena 130167 | CANDIDATE |
| flush trench N80 | 0.8996 | +0.0132 | — | 1392 | 1558.066 | 15.47 | Athena 128918 / 130184 | CANDIDATE |
| air-cladding comb r141 δx133 (**mechanism study only — the device stays SiN**) | 0.9007 | +0.0143 | 0.0964 (−12.2%) | 1340 | −26 pm | 15.44 | IGUM **52391** row 0 | CANDIDATE; user rule: never headline as a device |

### 3.5 Cavity-loss program bests (accurate mesh dx ≈ 35 nm, window 1558.5/40 nm/3001 pts)

Scope was fixed by the user: modify ONLY the cavity segment + 1–2 adjacent teeth, with
|Δfwhm| ≤ ~1%. Control loss 0.1174, T 0.878, λ_res 1556.1 nm.

| device | loss | T | Δfwhm | job | status |
|---|---|---|---|---|---|
| plain **rect-1050** cavity | 0.0823 (−29.9%) | 0.917 | +0.62% | Athena **117784** | accurate-mesh confirmed; robust fallback |
| **rect-1050 + inner see-saw δ+20** (teeth ±1 = 1020, ±2 = 980) | **0.0810 (−31%)** | **0.9179** | +0.8%, λ unmoved | Athena **117814** | accurate mesh; dose-response across 4 points = internal validation |
| **W1050 + inner gap-shift pair [+20,+20]** (lengthen_cavity on) | 0.0549 | 0.9444 | +1.0% (at bound), Q 1403, λ 1556.58 | Athena **117927** | accurate mesh |
| **the stack: W1050 + pair[+20,+20] + see-saw (1040, 980)** | **0.0545** | **0.9449** | +0.9% | Athena **118214** | accurate mesh; proven LOCAL OPTIMUM of the local-boundary space (job 118473 derived-profile falsification: both signs worsen) |
| (off-bound reference) triple 20×3 | 0.0403 | 0.9591 | +2.9% | 118214 | out of width bound |

Width ladder at accurate mesh is FLAT 1040–1075 (1050 → 0.0823, 1052 → 0.0821 = the 2e-4 in-study jitter floor) ⇒ the cavity-width knob is exhausted. See-saw is antisymmetric, saturating (δ = +20 and +30 both 0.0810) and linear through zero ⇒ genuine interference cancellation; the opposite sign hurts ×3.5. Best-device layout: `results_from_athena/tm_shift_frontier/layouts/layout_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8.fsp`. Frontier law MEASURED: **−1e-3 loss per +0.1% fwhm**, one dose parameter for singles/pairs/triples/length. Modularity **sign-inverts under apodization** (the pair: −27.4e-3 standalone → +26.2e-3 under apod-10) ⇒ combined designs must co-optimize.

### 3.6 Cross-cutting comparability traps (read before quoting any width or λ)

- **Two meshers in one repo.** `bragg_device.py:780` = conformal variant 0 (every SweepSpec study, all numbers above); `runners/lumopt2_design/lumopt2_design.py:745` = precise volume average (every lumopt2 campaign number). Same nominal N=100 corr-325 device: λ 1564.276 (PVA) vs 1559.006 (conformal) = **+5.27 nm**; mode FWHM 17.7005 vs 19.2448 µm = **−8.0%**. Rough conversion PVA ≈ 0.92 × conformal. 2026-08-21 research digest: presume **conformal variant 0** is the better absolute reference; PVA is a gradient-smoothness tool. The ~20 µm spec is conformal-defined.
- **`spectral_fwhm_nm` ≠ `fwhm_m`.** Q = λ_res / |`spectral_fwhm_nm`| (stored NEGATIVE, take abs; `post_processing.py:70`). `fwhm_m` = SPATIAL FWHM of the |E|² envelope along x (`post_processing.py:79`) — the mode width in the spec.
- **Right-arm shift index convention differs** between `bragg_device.py:1180` (`s_prev = shift_for_tooth[d-1]`) and lumopt2's `make_func` (`s = shift[i]`): 75 mismatching properties, all on the right arm, up to 6.43 nm displacement on `BEST_T9636`. Do not rebuild an optimizer device through SweepSpec without a scene diff.
- **Accurate-mesh λ offset** is real and not a bug: corr-400 N=80 reads λ 1555.95 (accurate) vs 1558.6 (optimization). Never compare λ across mesh modes.
- **Still CANDIDATE (no accurate-mesh two-step):** the 2026-09-12 comb-pair and 61-post q3db devices; comb N=41/47 at N=80; narrow-touch; air-cladding comb; flush trench (accurate confirm PARKED); the two −3 dB trench operating points (accurate validation PARKED); TE periodic comb 270° (+0.0033 = 1.8× floor). The only §2-two-step-CONFIRMED decoration is the 31-post comb at N=80 (job 130154).

---

## 4. Physics models and predictive tools

## 4.1 The q3db predictive engine — DESIGN-GRADE for bare uniform gratings

**Claim.** Predict T, λ, Q_L, spectral + spatial FWHM of a long pi-shift grating without simulating, and design a device at any dB point / any mode width with **ONE** confirmation run instead of a tuning ladder.

**Model.** L0 exact two-port algebra (`Q_c = Q_L/√T`, `Q_i = Q_L/(1−√T)`, `T = (Q_i/(Q_i+Q_c))²`) + per-family exponential Q_c(N) + **saturating** Q_i power law `1/Q_i = 1/(A·N^p) + 1/Q_sat` + a width-truncation fit. Measured saturations: bare c325 ≈ 48k, c276 ≈ 81k, invdesign ≈ 341k, itai_tm ≈ 126k, itai_te ≈ 717k, TE c250 flat ≈ 43k; all families p ≈ 2.9–4.4 **with** saturation.

**Files & exact invocation**
- `python_tools/predict_q3db.py` — **no CLI args** (CLAUDE.md §11): edit the knobs at the top and run `python python_tools/predict_q3db.py`. Knobs: `MODE` ∈ `"observe" | "design" | "extend" | "compare"`, `FAMILY`, `N`, `TARGET_DB` (default −3.0), `TARGET_WIDTH_UM` (e.g. 14.0; `None` = keep corr), `ROW` (the new measured device), `BASE_FAMILY`, `MEASURED`, `COMPARE_FROM_ROW`.
- `python python_tools/calibrate_q3db.py` from the repo root = **THE verification** (runs backtests B1–B14 to stdout); `python calibrate_q3db.py --csv out` also rewrites `python_tools/q3db_calibration.csv`.
- Skill: `.claude/skills/predict-q3db/SKILL.md`; self-contained state `python_tools/Q3DB_PREDICTOR_HANDOFF.md`; running state memory `project_q3db_predictive_engine.md`.

**Calibration data** (stored `.mat` only, zero GPU) — `DIRS` in `calibrate_q3db.py`: `results_from_igum/tm_nladder_c325/results`, `results_from_igum/trench_q3db_20um/results`, `results_from_igum/invdesign_q3db_20um/results`, `results_from_athena/invdesign_q3db_20um/results`, `results_from_athena/tm_te_apod/results`, `results_from_athena/te_q3db_20um/results`, `results_from_igum/tm_nladder_c276/results`, `results_from_igum/tm_q3db_14um_knob/results`, plus `results_from_igum/itai_hh_summary.csv`. Output table `python_tools/q3db_calibration.csv` (families: `errband`, `itai_te`, `itai_tm`, `knob_te`, `knob_tm`, `te_q3db_c250`, `tm_bare_c276`, `tm_bare_c325`, `tm_bare_c448`, `tm_invdesign`, `tm_trench_c325`). Fixed constants inside: `SIG_T = 0.0018`, `SIG_LNQ_LO = 0.01`, `SIG_LNQ_HI = 0.20`, `Q_ADEQ = 5e4`, `KAPPA_ANCHORS = [(325.0, 0.0353e6), (400.0, 0.0440e6)]`; in `predict_q3db.py`: `CORR_QI_EXP_TM = -2.9`, `FINF_CORR_EXP = -1.11`, `QC_H_PER_NM = -0.002818`, `CORR_BOUNDS_NM = (150.0, 650.0)`, `N_SEARCH = (20.0, 3000.0)`, `CORR_MATCH_NM = 2.0`, `NUMERICS_NOTE = "y8.0/z8.8 box, 20nm window/4001pts (3nm/0.75pm if Q_L>5e4), dx50 conformal, ASL 1e-7"`.

**Validated accuracy.** Backtest history: 37/39 → 39/41 → **44/46** gated (2 deliberate stress FAILs) → **48/50 gated** after the 2026-09-11 audit (commits 6126526 + 9b8de59). Gated hold-out Q_L errors 0.7–7.0%, median ≈ 2.7%. Flagship B1b: fit invdesign N=100–200, predict held-out N=220 → Q_L +2.3%, T −1.4 pt, crossing −0.3% (measured crossing N=220, Q_L = 88,868). B13 TE hold-out: fitted on N=166–190, predicts N=215 to Q_L +2.2% / T +1.7%. B7 (>1e5 regime, Itai TE): Q_i to 4.7–5.6%. Design-mode self-check for bare_c325: N=164 / T 0.4996 / Q_L 13,495 / width 19.97 vs ladder-measured N=165 / 0.4906 / 13,930 / 19.97. **Empirical deviation bands** (what a landed run is judged against; CSV family `errband`): span ≤30 periods → Q_L ±3.2% / T ±0.007 (12 rows); 31–45 → ±5.2% / 0.005 (6 rows); >45 → ±6.6% / 0.017 (2 rows, max not p90). Corr-moved designs quote ±10% / ±0.03.

**Live validations (5).** (1) IGUM **67731**, c276 N=200, 35 periods beyond its calibration: T 0.5696 (pred 0.5641, band 0.5257–0.5998), Q_L 19234 vs 19599 = **−1.9%**, width 23.91 = −0.1%, λ 1559.92 = −0.01 nm → PASS both bands. (2) IGUM **68086**, one-shot corr-448 14 µm design: **MIXED** — T 0.5808 above band, Q_L −16.9%, width **+2.1%** (knob worked), Q_i 16197 vs 15600 = +3.8% (corr^−2.9 validated), all the error in Q_c (5056 vs 6602). (3) IGUM **68925**, re-designed with the two-term fix: T 0.5097 vs 0.512, Q_L 4644 vs 4666 = −0.5%, width 14.19 = +0.6%, λ +0.05 nm → **PASS on all four**. (4)+(5) Athena **146681** comb pair N=172 and single 61-post N=171: engine misses ≤ 0.4% Q.

**Scope of validity.**
- Bare uniform gratings — TM corr 276/325/448, TE corr 250: **DESIGN-GRADE**.
- Decorated (trench / flush / comb): only via measured Q_i multipliers at the −3 dB anchor (backtest B8) + the `tm_trench_c325` family; EXPECTED-grade elsewhere.
- Inverse-designed device: `tm_invdesign` AS MEASURED only; any other shift/comb setting needs `extend` mode with its own anchor row.
- Apodized: **WIDTH** via the CMT κ(z) engine (B11 TM 0.4–0.9%; B11-TE +2.0/+1.1/−1.4/−4.8%); T and Q only as `itai_*` shapes.
- **Tooth shifts: NOT modeled** (a shift is a phase perturbation, not a κ change).
- The TE corr knob rests on ONE N=80 legacy point + TM exponents: EXPECTED-grade.
- Calibration/anchor device must satisfy **2κL ≳ 3.2** (c325 ⇒ N ≳ 93) — below that the prediction is refused. T ±0.03 holds ~30 periods beyond the anchored range; beyond ~45 it is a band (boundary B2-E, kept FAILing on purpose).
- "Extending" = adding **UNIFORM periods outside** (user rule 2026-09-11). Whatever is inside (apodization, comb, shifts) is carried only by the anchored levels; `ROW` corr = OUTER corr; the inside is not modeled.

**Known failure modes.** Extrapolating **ln T** instead of Q_c (missed a crossing by +191%); a pure power law through the Q_i knee (gives p = 0.73 vs true ~3.2 + saturation); calibrating κ on a **Q level** (ill-conditioned by A, §4.3 — use widths, which are box-independent, or the Q_c growth between two rows); a knob transform with the **rate only** (κ ∝ corr) and no LEVEL/intercept term (put Q_c +31% off); mixing polarizations or meshers (the tool now refuses); a dB target no length reaches (refused in one line). Standing rule: **decompose every miss into Q_c and Q_i before touching the model**, and check any new knob transform reproduces the STORED ladder in that knob at fixed N (free). Rule of thumb on anchoring: 2 rungs pin N*; a 3rd within ~40 periods pins Q to ~7%; a 4th to ~2–3% (walk-forward: +14.2% → +6.7% → +2.3%).

## 4.2 `bragg_cmt.py` — the CMT / TMM shape engine

**Claim.** 1D piecewise coupled-mode transfer matrices (Erdogan, *Fiber grating spectra*, JLT **15**, 1277 (1997)) in the Bragg-synchronous frame, supporting κ(z) apodization, π / fractional phase plates, z-dependent complex loss, and mode envelopes — the fixed-N
**SHAPE** tool of the q3db engine.

**Invocation.** `python python_tools/bragg_cmt.py` runs the self-test gate suite (prints `bragg_cmt selftest: all gates pass (G5 Qc ratio … vs …; G6 env FWHM … um)`; G7 = reciprocity of the flipped asymmetric device). **Run it after ANY edit to the file.**

**Conventions** (all gated): z increases left→right; R = forward, S = backward; `delta(λ) = 2π·n_eff/λ − π/Λ`; `sigma` complex per-segment DC term, `Im(sigma) > 0` = LOSS; `M_total = M_n @ … @ M_1`; `r = −M21/M22`, `t = det(M)/M22`; the π shift is a physical extra half-period of unshaped waveguide with plate phase `φ = (delta+sigma)·Lc + π·Lc/Λ`.

**Accuracy / scope.** Apodized mode widths **<1%** (backtest B11, rows A2–A20, κ anchored per family on ONE A0 width) — the case the deleted closed-form width law could not do; TE apodized widths +2.0/+1.1/−1.4/−4.8% (inside the 5% gate, but a band). Spectral-fit lane B10: fitting κ, n_eff on ONE short rung's stored spectrum gives λ to 0.01 nm and Q_c +6.7% at N=165 — **mask the resonance notch** (Rahimof-style window otherwise), and use engine Q_c only as a **shape ratio anchored on a measured row** (absorbs the n_g/n_eff level offset). Validity: κΛ ≈ 0.02 ≪ 1, first-order Bragg, **no dispersion (n_g ≡ n_eff)**.

**Known failure modes.** Engine width-vs-N is flatter than measured and its crossover Q_c steeper (INFO rows B2-C / B5-C) ⇒ N-trends go through the empirical fits, never the engine. Constant-α CMT loss cannot represent the measured envelope-limited Q_i ∝ N^3+ with saturation — radiation is injected as its own per-family saturating law, never fitted inside CMT. The legacy MATLAB @BraggGrating engine has known bugs (dz sign flips loss→gain, S↔T convention corrupting the reflection-phase correction, T>1 at the CMT↔FDTD seam patched by a one-sided `junctionEta` fudge, 4 inconsistent spatial-FWHM definitions) — this is a fresh reimplementation, not a port of those paths.

**Authorization.** CMT is **authorized for the q3db program** (user, 2026-08-31). The CMT ban is scoped to the **lumopt2 optimizer / width-wall** context only — do not reintroduce a CMT width model there (a coupled-mode-theory width model was tried, was wrong, and was deleted by user order).

## 4.3 The Q3dB measurement method (how to get Q at −3 dB cheaply)

**Exact algebra** (symmetric two-port, on resonance): `1/Q_L = 1/Q_i + 1/Q_c`, `√T = Q_L/Q_c`, `Q_i = Q_L/(1−√T)` ⇒ **`Q(−3 dB) = (1 − √0.5)·Q_i = 0.29289·Q_i`**, identically. The only physical assumption is that Q_i is the same at the operating N as where it was measured.

**Validated on our own four directly-measured T ≈ 0.5 anchors (MEASURED, ~2%):**

| device | measured T | measured Q_L | 0.2929·Q_i | err |
|---|---|---|---|---|
| TE N166 corr250 | 0.4919 | 12903 | 12655 | −1.9% |
| TM N165 no-trench | 0.4910 | 13930 | 13632 | −2.1% |
| TM N170 trench | 0.5021 | 18777 | 18873 | +0.5% |
| TM N169 trench bracket | 0.5130 | 18279 | 18867 | +3.2% |

Consequence: a 17 h crossing run buys ~2% over a 3 h row at T ≈ 0.88. If the deliverable is a RATIO against a stored anchor, do not pay for the crossing.

**Extrapolate Q_c, NEVER ln T.** MEASURED (te_q3db_20um): `dlnT/dN` = −0.0058/period at corr 233 but **−0.0426/period** near the crossing at corr 250, and the corr-233 slope did not transfer. `Q_c(N) = Q_c(N0)·exp(2·κ_bulk·pitch·(N−N0))` is linear in N by construction; get Q_c per row as `Q_c = Q_L/√T`. Two rows 32 periods apart pin the growth rate to ~1%, which moves N* by **0.3 periods** — N* is limited by Q_i, not by the slope.

**Conditioning** `dQ_i/Q_i = dQ_L/Q_L + A(T)·dT/T`, **`A = √T / (2(1−√T))`**:

| peak T | 0.975 | 0.95 | 0.90 | 0.80 | 0.70 | 0.60 | 0.50 |
|---|---|---|---|---|---|---|---|
| A | 39.3 | 21.7 | 9.2 | 4.7 | 3.0 | 2.2 | 1.7 |

Never quote Q_i from a T > 0.95 row without saying so; T = 0.7 → 0.5 buys only 1.8× in conditioning at ~3× the wall time.

**Known failure modes.** (a) **Q_i drifts below the operating point:** MEASURED 58k → 76k over N = 110 → 165 at corr 276 (**+31%**), while te_q3db_20um's Q_i was flat within 4% over N = 166–215 — Q_i keeps moving while the device still truncates the mode's tails (24% of ∫x²I beyond the device end at N=110, 6% at N=165). "Measure Q_i cheap at low N and multiply by 0.293" is unsafe until saturation is DEMONSTRATED by two Q_i values at different N agreeing. (b) **The containment threshold does NOT transfer (FALSIFIED 2026-08-26):** Itai's Nt60 TE device sat past the TM study's saturation benchmark (end-field 5.5e-3 at N=98, 2.0e-3 at N=130) and Q_i still went **610k → 1,159k (+90%)** (jobs 63722 / 63752, both fully gated). Containment is device-specific — it depends on the apodization envelope's spatial-frequency content inside the light cone. (c) **Cost blows up at the answer:** `t = k·N·(t0 + 16.1·τ)`, `k = 0.00249 min/(N·ps)`, `t0 = 66 ps`, `τ = Q·λ/(2πc)`; worked (Itai Nt60 TE): N=130 → 2.9 h, N=150 → 7.8 h, N=160 → 12.2 h,
**N=168 (T=0.498) → 17.1 h**, N=172 → 20.0 h, nothing past ~N=175 inside a 23:30 wall. Athena's contended nodes are **1.67× slower** than IGUM's A100s.

**The recipe (2 simulations beyond the first).** (1) one cheap row at the design's own length → Q_L, T, Q_c, Q_i; (2) one row ~30 periods longer, still cheap (T ≈ 0.85–0.9) → Q_c growth rate to ~1% + a second Q_i at better conditioning = the saturation check; (3) solve for N* at T = 0.5 with the exact algebra and run ONE row there. If step 2's Q_i disagrees with step 1's by more than a few %, discard the low-N value.
**Per-row gates:** resonance inside the window · T above the dead floor (dead device T ≈ 0.0008) · **T + R < 1** · `fwhm_m < 0.45·L_device` · **≥ 10 points per linewidth** · accept T = 0.5 ± 0.03.

## 4.4 The comb k-space model (`python_tools/comb_kspace_model.py`)

**Claim.** One semi-analytic model that reads the device's radiation leak from stored near-field planes (unclipped, unlike the far-field monitors), models the cladding post comb as a row of z-dipoles driven by the measured cladding tail, predicts ΔT for every stored comb row, and only then designs the next comb. Zero GPU. Needle = the exact Lorentzian tail of the measured envelope (κ 0.040/µm, c±from the axis field) on a fine kx grid; comb = array factor with mirror rows `2cos(k_perp·d·cos ψ)`.

**Invocation.** `python python_tools/comb_kspace_model.py` (no CLI args) runs the `__main__` diagnostic: loads the control planes and prints `ctrl planes: T … slice … nm (res …), fwhm … um`, the x-grid uniformity check, then the leak spectra. Design/fit entry points used from scratch drivers: `fit(...)`, `design_scan(planes, alpha, f, b, kxf, L, Lambdas=(524,526,528,530,531,532,534,536), phases_deg=range(0,360,15), Ns=(21,31,41,51,61,81,101), ds=(1.0,1.2,1.4,1.6,1.8,2.1))`, `best_for_geometry(...)`, `geometry_row(Lambda_nm, dx_nm, n_posts, d_um, r_nm=110.0)`. Constants: `LAM_NM = 1558.61`, `N_CLAD, N_CORE = 1.444, 1.97`, `CTRL_T, CTRL_LOSS = 0.8851, 0.1110`, `FLOOR = 0.0018`, `R_MIN_NM, R_MAX_NM = 55.0, 150.0`, `BETA = 1.5078 * K0`, device length 83.0 µm.

**Calibration data.** `results_from_athena/comb_physics_rethink/data/` — 92 far-field `.npz` + `manifest.csv`, plus 3 `*_PLANES_RES.npz` resonance slices of the `scat_i_fieldmaps` 3D planes (job **123991**); control planes `result_N80_TM_avg_Ybox16p0_Zbox8p8_PLANES_RES.npz`. Calibration studies `scat_p_antineedle`, `scat_r_aim536`; validation set adds `scat_s_refine`, `scat_w_dscan`, `scat_y_polish`, …

**Validated accuracy.** First fit on 13 rows (2 phase circles + d-scan + N47/61):
|α| 0.0103, arg −82°, f 59, b → ∞ (the needle behaves **top-going** — no phase rotation
with d in the T data). Blind: 37 cladding rows **rms 0.0049, 27/37 within 2× the 0.0018 floor**; reproduces Λ 530–540 @270°, the r-scan, N 41/53, the 2-row / 4-row nulls, radius-apodization, d 2.1. Refit on 16 circle rows (after the off-centre campaign):
**rms 0.0034**. Blind corr-325 transfer check: predicted 0.4391 vs measured 0.4371.

**Known failure modes — the model is RANK-ONLY.** It **over-predicts, and its optimism grows with post count**: +0.004 at 31 posts, +0.008 at 62, +0.018 at 122 (largest miss: predicted +0.039 for two 61-post combs, measured a tie at +0.0206). Never extrapolate to bigger arrays or outside the validated amplitude range (a Born extrapolation gave +0.10 at d = 1.0 with an unphysical f = 59, and every stored strong-drive row under-performs it: d1.5/r92 measured +0.0130 vs +0.0186; air r141 +0.0143 vs +0.034). It gets the dx = 0 trend wrong at Λ ≥ 548, over-predicts d = 1.5 and air r141 by ~2×, and for **in-core** structures it is first Born only (P_c is 5–16× the leak ⇒ dead, no number quoted). The complex-amplitude pattern search was itself wrong once (`a·C(dx=0)` is not the dx knob when m = 0 end-scattering matters) and was replaced by a physical (centre, dx, n_posts) grid + Gram-matrix quadratic form.

**Physics it established (measured, not modeled).** The comb's cost follows its **END positions**, with sign period `π/(β−k_c)` = **12.2 µm**; the winning pattern is one comb per side at ±12 µm, each at 90° (not the centred 270°), or equivalently ONE centred 61-post comb with ends at ±16 µm; never two combs on one centre. Phase convention: `φ = 360·δx/Λ` measured from the **CAVITY CENTRE**. Family saturates at ends ±16–19 µm; third combs, fan combs, extra rows, curves, chirps, clusters and rod-shaped posts are all ruled out (model and, where run, measurement). Comb and apodization are **alternative** needle-killers, never stackable (apod-10 + comb −0.0047; + pair −0.0062); the trench is the only measured apod-compatible add-on (+0.0039). Handoff: `runners/scatterers/COMB_HANDOFF.md`; writeup `docs/comb_physics_rethink_2026-09-11.md`.

## 4.5 The target-locking method (`lock-target` skill)

**Claim.** Hit EXACT targets (spatial mode width, peak T / dB point, resonance λ, Q via a loss knob) with a knob table and a solve order — not an optimizer.

**Knob table** (`.claude/skills/lock-target/SKILL.md`; memory `project_target_locking_method.md` is the pointer):

| target | knob | linearizing coordinate | side effects |
|---|---|---|---|
| spatial mode width `fwhm_m` | corrugation depth | 1/FWHM vs corr (FWHM = ln2/κ, κ ∝ corr) | changes T and Q strongly; retune N after |
| peak T (e.g. −3 dB) | `n_periods_each_side` | ln(T) vs N (locally linear) | width shifts only ~4% over ±30% N; λ unmoved |
| resonance λ | pitch | λ vs pitch (linear) | negligible; Δλ ≤ 1 nm acceptance |
| loaded Q at fixed T | NOT free: `Q_L = (1−√T)·Q_i` | — | needs a LOSS knob (trench, apod) as an extra dimension |

The system is nearly **triangular**: solve **width → T → λ trim**. Protocol: predict → one zipped `SweepSpec` ladder of 3–5 bracketing points as ONE array → fit in the linearizing coordinate → ONE integer confirm (regula falsi if it misses; never redo the ladder). Tolerances (physical floors): width ±1 µm (±0.25 on request), peak T ±0.03 (integer N quantizes T by 0.01–0.02 per period near T = 0.5), Δλ ≤ 1 nm.

**Validated.** trench_q3db_20um: two exact targets locked in **2.5 rounds / 29 sims** (corr fit corr(20 µm) = 324.7, residual 0.17 µm). Sibling shortcut on te_q3db_20um: the same two targets in **9 sims** by riding the measured line shapes with 2-point lines.

**Known failure modes.** (a) A T(N) line does **not** transfer across corrugation — `dlnT/dcorr` ≈ −0.05/nm at fixed N (corr 233 → 250 collapsed T 0.58 → 0.26); re-anchor T after every corr move. (b) Legacy anchors mislead: corr 300 = 19.1 µm in old data but 21.5 µm in-study → an 8-sim hedge ladder ran at the wrong corrugation. Calibrate ONLY from in-study points at identical numerics. (c) Q_i drifts with N → measure near the operating point. (d) Q is only reportable with **≥ 10 sample points across the spectral linewidth**; under-resolved points are excluded, not reported. (e) Filename collisions: at W800 the corrugation enters `sim_helpers.generate_file_tag` only via the TM `_C{corr}` branch — verify tag uniqueness before dispatch. Auto-shutoff is SETTLED at **1e-7** for everything (`cfg.mesh.auto_shutoff_min`, None = 1e-7); 1e-6 already costs 2% Q at Q ≈ 1.5k and 16% at Q ≈ 27k; 1e-8 is unreachable (total-field energy floor ~5e-8).

## 4.6 TM radiation design rules (memory `project_tm_radiation_design_rules.md`)

**Claim.** A first-order grating cannot radiate (at Bragg, every order sits at
|k| ≥ n_eff·k0 > n_clad·k0); radiation comes ONLY from where periodicity is broken — the
defect — and its strength is the mode envelope's Fourier weight inside the cladding light cone. Therefore loss is **envelope-limited** and envelope engineering is the right axis.

**MEASURED exponents** (free, from stored `.mat` in `tm_nladder_c325/`, `tm_nladder_c400/`): N-ladder at corr 325 (5 points, same box) → **Q_i ∝ L^3.60**; corrugation 325→400 at N=60 → L^2.45; at N=70 → L^2.58. Brackets the theoretical L³ for an exponential (cusped) envelope; **no dominant distributed loss floor**. Caveat: Q_i = Q_L/(1−√T) is stiff near T→1 (±11% at T ≈ 0.91) so the exponents carry ±0.3–0.5. Q ∝ L² is wrong.

**Mode length saturates — L was never a lever.** N = 100 → 120 grows the mode only
**2.2%** (19.245 → 19.661 µm) while T falls 0.9104 → 0.8441; corr-325's mirror-limited asymptote is **~19.7–20 µm**, i.e. the ~20 µm spec IS this family's natural mode length. At N ≥ 100 only the envelope **SHAPE at fixed L** remains.

**Why TM ≠ TE (three independent reasons).** Half the k-space margin: `δk = (n_eff − n_clad)k0` = **0.507** (TE) vs **0.258** rad/µm (TM-anchored) ⇒ smoothing length 1/δk = 1.97 vs 3.87 µm. The TM bandgap collapses in thin cores (Zhang/McCutcheon/ Burgess/Lončar, Opt. Lett. **34**, 2694 (2009): Q_TM 2.4e6 → 9,000, ~270×, from 3:1 to 1:1 thickness:width; our core is 350×800 = 1:2.3). And the tooth shift is THREE perturbations — phase + duty cycle + **DC index** (receipt: λ +1.6 nm per +374 nm of 2Σs) whose DC term scales as 1/(n_eff − n_clad), ~2× larger for TM ⇒ the TE/TM shift comparison is confounded.

**The design rule that works — the INNER SEE-SAW.** teeth ±1 = 1000+δ, ±2 = 1000−δ (job 117814, accurate mesh): loss −31%, T 0.878 → 0.9179, fwhm +0.8%, λ UNMOVED. Four properties, all required: LOCALIZED · ZERO NET AREA · ANTISYMMETRIC · TRANSVERSE (width, not segment length). It is pure corrugation and was in the optimizer's basis all along (`Δcorr_d = ±δ`, `Δavg_d = ±δ/2`; on corr-325/W800 with δ=20: tooth 1 → corr 345/avg 810, tooth 2 → corr 305/avg 790) — **the optimizer never used it** (Johnson 2001: the Q peak is a sharp Lorentzian in parameter space). **Seed the see-saw; do not expect to discover it.**

**The other width-neutral lever is the comb** (corr-325 N=165, clean with/without control): width 19.9702 → 19.9001 µm (−0.35%), **Q_i 46,499 → 54,457 (+17.1%)**, T +0.046; a mis-placed comb variant drops Q_i to 38,784 (below control) — the strongest evidence for coherent interference rather than a bulk effect.

**Scope / open edges.** Cavity work is capped and largely spent (round-7 k-space diagnostic: only ~30% of radiating weight is cavity-local and the stack already harvests ≈ that; the remaining ~70% is distributed along the arms). CLOSED, do not re-propose: distributed π-shift · step-envelope islands · inner-tooth shapes · wall-phase offset · anti-radiator asym-DW · hourglass · external scatterers · cavity SHAPE on top of rect-1050 · **moiré** (one beat node ⇒ mode FWHM ≥ 50.8 µm, 2.5× too wide) ·
**x-asymmetry** (antisymmetric perturbation → odd δA ⊥ even A₀ ⇒ strictly adds radiation; the comb is NOT a counter-example — it is a separate far-field radiator). Note the internal disagreement, now resolved: this memory records `κ ∝ corr does NOT hold 325→400` with Q_i ∝ corr^−1.8; the q3db engine's later reconciliation (B12) says κ ∝ corr IS solid for the **coherent** channel (0.1–1.3% over 276/325/400) and −1.8 was a mixed-N fit, the true **radiative** law being Q_i ∝ corr^−2.90 at fixed N=150. Use the latter. DERIVED, not measured: apodization has a crossover at FWHM ≈ 14 µm below which it HURTS.

## 4.7 Measured width-reducing lever inventory (TM)

Answer to "did anything ever reduce TM mode width?" — four effects, each compared to **its own in-study control**, never across studies (memory `project_tm_width_reducing_levers.md`).

| effect | study dir | Δwidth | ΔT | verdict |
|---|---|---|---|---|
| **air trench** (rect L84 µm × W800) | `air_trench_w1050` | **−0.85%** | **+0.0157** | WIN-WIN |
| **cavity width W1250** | `cavity_width_ladder` | **−0.29%** | **+0.0178** | WIN-WIN |
| cavity width W1400 | `cavity_width_ladder` | −0.74% | −0.0061 | ~T-neutral |
| cavity **hourglass** pinch 150 | `inner_shape_study` | **−1.05%** | −0.0280 | narrows, costs T |
| cavity hourglass pinch 75 | `inner_shape_study` | −0.50% | −0.0147 | narrows, costs T |
| **comb** (corr-325 N165) | `comb_q3db` | **−0.35%** | Q_i +17.1% | WIN-WIN |

Controls: `cavity_width_ladder` + `inner_shape_study` → in-dir `result_N80_TM_avg_Ybox6p8_Zbox8p8.mat` (15.532 µm, T 0.8864); `air_trench_w1050` → in-dir `..._ff.mat` (15.622 µm, T 0.9218).

**The pattern.** Narrowing lives in the **CAVITY** (where the mode peaks) or in an **added scatterer** (trench, comb) — **never in the teeth**. Apodization, tooth shifts (both duty-cycle signs, both segments), tooth shapes (ellipse/tri/wedge: +1.6% to +9.8%) and the see-saw all WIDEN. The no-go: envelope ~ exp(−∫q dx) with q = √(κ²−δ²) ≤ κ, so any tooth-level detuning only lengthens the decay; narrowing needs κ raised near the CENTRE. Hourglass (pinch) narrows / barrel (bulge) widens — a clean antisymmetric pair (`bragg_device.py:1154-1157`: cavity drawn as `w = W_cavity ± depth·sin(πu)`; full ladder vs rect control 15.532 µm / 0.8864 / 1558.62: hour150 15.370/0.8583/1558.50, hour75 15.454/0.8717/1558.55, barr75 15.585/0.8971/1558.66, barr150 15.622/0.9069/1558.70). Cavity width is NON-monotonic: W1050 +0.59%, W1150 +0.18%, W1250 −0.29%, W1400 −0.74%.
**Key distinction:** hourglass/barrel move you ALONG the usual TM trade curve; only the
**air trench** and **cavity W1250** fall OFF it (narrower AND higher T).

**Caveats — do not overstate.** All of these are corr-400 N=80 (~15.5 µm modes), not the corr-325 ~20 µm production family; the comb row is the only corr-325 datapoint, so porting is UNVERIFIED. All are ≤1% while the campaign's problem was +15%: they are counterweights, not a solution. Stacking is unmeasured, and modularity in this program has already sign-inverted once under apodization. The width jitter floor at dx = 50 nm was never measured for this family (the 0.03% on record is BOX-size variation at corr-325 N100, PVA), so the −0.29% row is the one most likely to be noise. Raising the **uniform** corrugation also narrows but is excluded by user rule; inner-region corrugation SHAPE remains fair game.

## 4.8 ★The measurement-adequacy trap above Q ≈ 5×10⁴

Above Q_L ≈ 5e4 the q3db family's standard recipe stops being adequate in **two independent ways**. Neither announces itself; **both bias peak T LOW**, which drags an apparent −3 dB crossing to lower N and then looks internally consistent.

**(1) Spectral under-sampling.** The family window is 20 nm / 4001 pts = **5 pm/sample**; linewidth = λ/Q:

| Q_L | linewidth | samples across FWHM at 5 pm |
|---|---|---|
| 10 500 | 149 pm | 30 — fine |
| 53 000 | 29 pm | 5.9 — marginal |
| 143 000 | 11 pm | 2.2 — BROKEN |
| 241 000 | 6.5 pm | 1.3 — BROKEN |

At 1–2 samples the true peak falls BETWEEN grid points. **Fix:** keep 4001 points and NARROW the window onto the known resonance — 3 nm → 0.75 pm, 2 nm → 0.50 pm — keeping ≥ 1 nm of margin against λ drift with N.

**(2) Truncated ring-down** (the subtler one). `bragg_device.py:769-770` sets simulation time **2000 ps** (env override `TM_SIM_TIME_PS`) and auto-shutoff 1e-7; energy lifetime τ = Q/ω and the run needs **16.1·τ** to reach the shutoff:

| Q_L | τ | time to 1e-7 | fits 2000 ps? |
|---|---|---|---|
| 100 000 | 83 ps | 1 336 ps | yes |
| 143 000 | 118 ps | 1 910 ps | just |
| 174 000 | 144 ps | 2 324 ps | **NO** |
| 241 000 | 200 ps | 3 219 ps | **NO** |

A truncated ring-down convolves the Lorentzian with ripples of period λ²/(c·T_sim) = **4.06 pm** at 2000 ps — comparable to the linewidth itself, so it perturbs the half-max crossings, not just the peak. Residual field amplitude exp(−ω·T_sim/2Q) = 2.2e-4 at Q = 1.4e5 but 6.7e-3 at Q = 2.4e5 (~1.3% on T).
**Fix: `TM_SIM_TIME_PS=4000`** (residual 4.5e-5 even at Q = 2.4e5), at ~2× runtime.

**★The knob is NOT reachable from the sweep path.** `athena/deploy_athena.sh:987` (single-run) forwards `TM_SIM_TIME_PS` in its sbatch `--export`, but line **1256** (the `--option3` sweep path) does NOT, and its only hook `EXTRA_EXPORT` is reserved for `LOCKED_LAMBDA_FILE` (same on IGUM, `igum/deploy_igum.sh:1232`). For a SweepSpec study set `os.environ["TM_SIM_TIME_PS"]` at the **top of the runner module** — it is imported on the node (`athena_run_one.py:209`) before the scene is built — or add the variable to that `--export` list and mirror it to `igum/`.

**How to apply.** Before dispatching any rung expected above Q ≈ 5e4, compute BOTH (a) linewidth/grid and (b) 16.1·τ vs the configured simulation time, and state the numbers in the runner docstring. Cheap ladder rungs tolerate ~1% T error; the final quoted device does not. This also explains why a quoted Q_i of ~620k from a 9-points-per-linewidth row (§3.3) is a lower bound, not a measurement — and why cost scales with the very quantity the study maximises: against a 23:30 QOS wall, **N ≥ 240 cannot complete on Athena** (N=280 was cancelled at 9 h 23 m having needed ~40 h).


## 5. Inverse design (lumopt2) — method and state

Entry points, in reading order:

- `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\HANDOFF_2026-09-01.md` (241 lines) — the live state of the **d1 generation**; self-contained, carries the resume commands.
- `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\THEORY.md` (749 lines) — the METHOD (editable source).
- `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\HANDOFF_SELF_CONTAINED.md` (1089 lines) — THEORY + the full 191-param vector + code + raw data; hand THIS to a session without repo access.
- `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\HANDOFF.md` (3372 lines) — long operational log.
- Engine: `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\lumopt2_design.py`. Skill: `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\.claude\skills\lumopt2-design\SKILL.md`.
- Two tracks: **Track A** = the hand-steered parametric device (the deliverable). **Track B** = the adjoint inverse design (the multiplier). Nothing in Track B is required for the device to exist.

### 5.1 Objective and constraint, exactly

Maximise the resonance **peak transmission `t_pk`** of the corr-325 pi-shift Bragg grating (SiN `n_core = 1.97` / `n_clad = 1.444`, core height 350 nm, pitch 516.83 nm, TM, surrogate **N = 100/side**, frozen SiN comb) **while holding the envelope FWHM at 18.346 µm ±2%** — a hard, **two-sided** acousto-optic spec; a narrower mode is off-spec, not a bonus.

- The ride band at that target is **[17.979, 18.713] µm** (benchmark width 18.3460 µm, PVA). The band was made **symmetric ±2%** by user ruling 2026-08-23 (engine `RHO_DN` 0.95→0.98); an older era ran +2%/−5% (κ-ratio ρ deadband, β 18/5) and then +1%/−5% (`RHO_UP` 1.02→1.01, 2026-08-16).
- Transmission FOM — a **windowed power-mean (soft-max), p = 12**:

  `J_T = ( mean_{i in window} |T_i|^12 )^(1/12)`, window = `|λ_i − λ_pk| ≤ 2.5·FWHM`, re-selected every eval, with a stop-gradient and a dead-device raise.

  Soft-max not `max` because a hard maximum has zero gradient at every non-maximal sample and its argmax hops between grid points as the resonance drifts. **Windowed** because the global max of `T(λ)` sits in the passband, not at the defect resonance.
- Physics frame: `t_pk = (1 − Q_L/Q_i)²` with `Q_L ≈ 2000` pinned by the width spec ⇒ **every T gain is a Q_i (loss) gain**.
- Numerics of the campaign lane: PVA mesher, `dx = DX_PITCHLOCK_NM = PITCH/10 = 51.683 nm`, scan 10 nm / **501 points** (20 pm) — that is **40 points per spectral FWHM (810 pm)**, exactly adequate for the λ-stencil. **Do NOT widen `scan_width_nm` without raising `n_wl_points` in step.** Earlier eras: 301 pts @ 20 pm / ±3 nm, and ±5 nm / 501.
- `scan_center_nm = 1564.21` (PVA; +5.2 nm vs the conformal family).

### 5.2 The 191-parameter design vector

All values in **nm**. `N_FREE = 25` free periods per side (mirror-symmetric: one value drives both arms), `N_COMB = 57` posts. Slices and bounds from `param_bounds(spec)` (`lumopt2_design.py:507`):

| slice | n | meaning | bounds (nm) |
|---|---|---|---|
| `SL_CORR` 0:25 | 25 | corrugation depth per tooth — local κ; apodization lives here | `(150.0, spec.corr_max_nm)` (dip and overshoot both allowed; `corr_max_nm` default ~451, was ratcheted 451→429→407 by the legacy width-trip handler) |
| `SL_AVG` 25:50 | 25 | mean tooth width — local n_eff / detuning | `(775.0, 825.0)` = ±25 nm n_eff-drift cap |
| `SL_SHIFT` 50:75 | 25 | per-tooth longitudinal shift; cavity absorbs `2·Σshift` | `(0.0, 200.0)` — repo convention, **do not tighten**. If `freeze_shifts`: sliver bounds `(v−1e-3, v+1e-3)` around the evolved seed |
| `SL_R` 75:132 | 57 | comb post radii | `(70.0, 240.0)` when `free_comb` (70 ≈ the dx=50 mesh floor); otherwise sliver `(v−1e-3, v+1e-3)` |
| `SL_X` 132:189 | 57 | comb post positions | `(x−100.0, x+100.0)` about the seed; otherwise sliver |
| `I_DCOMB` 189 | 1 | comb transverse offset | `(1500.0, d_max)`, `d_max = box_y_um·1000/2 − 240 − 1200` (y-PML clearance) |
| `I_CAV` 190 | 1 | cavity width (transverse, y) — the most width-efficient lever measured | `(750.0, 1150.0)` |

- `N_PARAMS = 191`, asserted. On top of the box there is an optional **`spec.trust_nm`** per-block clamp (`corr`/`avg`/`shift`/`r`/`x`/`d`/`wcav`), asymmetric against the box since the 2026-08-24 fix (see §5.7 bug 9).
- Comb is **NOT x-mirrored**; the grating walk is mirror-symmetric and reproduces the builder to 0.0000 nm (gate B1).
- Outer periods beyond tooth 25 are pure mirror, frozen at corr 325; the surrogate `N` is chosen so the mirror is effectively infinite (`2κL ≳ 3.5`).
- Comb seed: 57 posts/side, Λ531 / δx401 / r80 / d1.9 µm. The count 57 was a **coverage** choice at N=100 extended from a MEASURED optimum of 47 posts at N=80 — never optimised at N=100 (a frozen uncertainty).
- **Freezing changes bounds, not geometry.** `free_comb=False` collapses comb bounds to ±0.001 nm; a seed or detune point that moves a frozen param is rejected outright by lumopt2 (`parametrization.py:674 _check_params`) and the job dies in ~60 s.

### 5.3 The two width measures, and the VOID window

- **`fwhm_env`** (`fwhm_env_of_line`) — **the spec observable**. Replicates `sim_helpers.extract_and_process_field_profile` step for step: pick λ_pk → `|Ex|²+|Ey|²+|Ez|²` → trapz over y → crop `|x| ≤ n·pitch` → `extract_envelope_peaks` (cubic through the standing-wave peaks) → `calculate_fwhm_relative` (half-max **relative to the floor**). Identical **by construction** to `post_processing`'s `fwhm_m`: validated to **7e-15 µm** on two stored devices (N100 c325 19.244767 µm, N80 c325 18.393528 µm); the re-derived envelope matches the stored `field_envelope_1D` to 5e-16 relative. **Not differentiable** (peak-picking + interpolation).
- **`softW`** — the differentiable carrier that drives the width adjoint: boxcar 258 nm + Gaussian 0.25 µm smoothing matrix → soft-max peak → fixed edge-window floor → sigmoid super-level indicator (ε = 0.05) → integrate. MEASURED tracking of `fwhm_env` growth: **≤1.6 pp (scipy form) / ≤2.2 pp (autograd form)** over 7 corrected campaign profiles spanning +4.9%→+26.6% true growth, ≤2.1 pp on the c400 shift ladder and the TE/TM shift families. Its autograd gradient equals the FD directional derivative to **1.4e-8**. softW is **LOCAL** (N-ladder-scale excursions err −8 pp) ⇒ it is tied to `fwhm_env` by a **delta anchor** re-measured at every accepted iterate, and the residual `wg_resid_um` is logged **every evaluation** with a loud warning if it drifts (`check_sigma_surrogate`, `:887`).
- ★ **All branch decisions are made on the MEASURED `fwhm_env`, never on the surrogate.** The surrogate only supplies a direction.
- ★★ **EVERY `sigma` AND EVERY `FWHM` LOGGED BY THIS ENGINE BEFORE 2026-08-18 IS VOID.** Cause: `profile_line()` never integrated over y — it flattened `(y, λ)` into one axis and indexed with the **λ index** (always `< n_lambda`), so it always returned **y-row 0**. The `field_profile` monitor is a 2D Z-normal plane of y-span `1.5·width_wide` (~1.5 µm), so row 0 sat ~0.75 µm off the guide axis in the evanescent skirt. Voided: every `sigma_um`, every FWHM, the 134217 audit rows, the sigma-neutral probe, the shift ladder's sigma values, and every sigma anchor/wall calibrated from them. **T / λ / Q_L / Q_i / R / loss are PORT quantities and are UNAFFECTED — they all stand.**
- Consequences that must not be re-cited: the trade line `T = 0.89265 + 0.01549·ΔFWHM`, the per-lever efficiencies 0.0203 / 0.0116 T/µm, "seedB ev1 is +0.0156 above the line", the FWHM_hat ratios 1.17–1.19, and the 3-point factorial slopes are ALL built on void widths — keep as a record of reasoning only.
- **Mesher discipline.** The campaign lane is **PVA**; the spec / q3db / Itai family is **conformal**. Same device: λ **+5.27 nm** (1564.276 PVA vs 1559.006 conformal) and FWHM **−8.0%** (17.7005 vs 19.2448). `BEST_T9636` reads **T 0.96361 PVA vs 0.97805 conformal**. **Never cross-quote.** PVA is used because under "conformal variant 0" the grid-aligned tooth edges give staircased dEps (tooth adjoint/FD 0.07–0.26, i.e. 4–14× too small) while comb cylinders were fine 0.77–0.98. The PVA-vs-conformal arbitration is PARKED.
- Width-metric noise floor: `fwhm_m` reads **19.2411–19.2471 µm across SEVEN boxes** for bare N=100 corr-325 = 0.006 µm = **0.03%** spread, so a 0.1% width change is resolvable. Over the same 7 boxes `T_res` spans 0.9091–0.9194 (0.010) — **T is ~30× more numerics-sensitive than FWHM.** T repeatability floor: **0.002** (gains < 0.004 are noise).

### 5.4 The projected-gradient algorithm

Why a single cost function provably cannot do this job (THEORY §4):

1. **σ (second moment) could not MEASURE the violation.** σ is tail/bulk-dominated; the spec is a half-max crossing. σ is **24 pp** off the true `fwhm_env` (participation ratio 21 pp) where softW is ≤2 pp. MEASURED: corrugation apodization alone moved FWHM **+4.89%** while σ moved **+0.001%** (17.2518→17.2520). Both design lineages went width-buying while the constraint reported healthy; the recorded best had grown **+14.9%** in true width. A fitted linear surrogate `σ̂ = 17.49 + 0.0051·(2Σshift) + 0.109·(w_cav − 800)` overstated corrugation's width authority by ~30% and falsely rejected an in-band design at T 0.9591.
2. **A fixed-μ penalty `J = J_T − μ·penalty(W)` could not PRICE it.** Three structural reasons, no μ fixes any: (a) a scalar fixes the exchange rate before you know the landscape (the logged shadow price varies by >1 order of magnitude); (b) a **deadband** penalty prices nothing INSIDE the band, so `∇J = ∇T` there and points reliably at a wider mode — monotone widening is what the formulation specifies, and two full campaigns ended out of band that way; (c) a scalar penalty can be blind to whole directions — the tooth-level wall was **rank-deficient**, pricing only `mean(corr)` (identical gradient on all 25 teeth) and only total elongation, leaving the **see-saw** direction unpriced (wall predicted −0.82 µm where MEASURED is −0.015 µm), ~48 of ~50 directions unpriced, `wcav` unpriced entirely.

⇒ **T and W stay SEPARATE objectives with SEPARATE gradients.**

**One iterate, start to finish:**

1. forward solve at `p` → `T(λ)`, field profile `I(x)`
2. measure `λ_pk`, spectral FWHM, `fwhm_env`
3. **two** adjoint solves → port-driven and width-driven fields
4. assembly pass 1 → `∇T`; assembly pass 2 → `∇W` (same fields, **zero extra solves**)
5. branch on the MEASURED width:
   - `W < target − margin/2` → **CLIMB**: `step = α·D·∇T`, UNPROJECTED, clipped so it lands exactly on the ceiling and never over
   - `W` within the margin → **RIDE**: the null-space step, `∇W·step = 0`
   - `W > target + margin/2` → **RESTORE**: straight back along `∇W`
6. clip to the bounds box; cap the step
7. accept/reject on a filter over (transmission, distance-to-target); on reject, re-step from the last accepted point at half the step length using its **STORED** gradients — no re-solve
8. log everything; persist for resume

**The projection.** `step = α·(D∇T − coef·D∇W)` with `coef` chosen so `∇W·step = 0` — a D-metric orthogonal projection of `∇T` into the null space of `∇W`. Verified numerically to **8.3e-17** in `gates/gate_projection_local.py`. The strategy is deliberately **ceiling-riding**: sit just under the maximum allowed width and spend the whole allowance on T.

★**Seed-dependent exception (MEASURED, b1 lane 2026-08-29):** a NEAR-CONVERGED seed must NOT inherit the ceiling-ride target. At `BEST_T9636` the width-blind climb bought **+0.00097 T for +0.272 µm = 0.0036 T/µm, 30× below the uniform lane's rate**, because ∇T there is aligned with the width direction. A best-seeded lane sets `wgp_target_um` = the seed's own `fwhm_env` so the constrained law engages from iterate 0.

**The ns2 two-constraint step (`_ns2_step`, the d1 generation).** Project `D·∇T` into the null space of **BOTH** the raw fixed-λ `∇W` **AND** `gλ = dλ_pk/dp`, with a Feppon range-space restoration folded into the same step (never stop-and-restore). Two measured defects it fixed:

- steps were a constant 10 nm: `_cap(a) = cap0·min(1, a/a0)` scaled the cap in lockstep with `alpha` while the step was ∝ alpha, so the delivered move was exactly `wgp_step_max_nm` whenever the raw step (~73 nm) exceeded it — independent of `alpha`, `wgp_step`, `‖∇T‖`. (This is why b2's step-doubling was a mathematical no-op.)
- width creep was **λ-slaving**: MEASURED `ΔW ≈ 0.3655·Δλ_pk` per iterate.

Key decomposition at BEST (toy 138658): `∇T` overlaps raw `∇W` by only **0.6%** (lam 0.0057) but `gλ` by **~85%** (rho_T 0.106–0.143) ⇒ **T rises mainly by red-shifting; the width creep of every old lane was the shadow of that drift.** Once `gλ·d = 0`, the fitted `wg_dwdlam = 0.3655` **cancels out of the feasible directions entirely** (gate-asserted for ANY coefficient); it survives only in diagnostics.

**Adaptive trust cap = real persisted state.** ×1.5 on a verified hold, halve on reject, floor 2 nm; stored in `<label>_optstate.json` so a REQUEUE/restart cannot reset it. MEASURED: gains scale ~linearly with cap, leaks ~quadratically; d1u eval 3 jumped `t_pk` 0.94680 → 0.95983 (**+0.0130 in ONE iterate at cap 33.75 nm**, 3.4× the 10-nm-cap rate) with λ slipping only +0.12 nm. **Both lanes broke at cap 60 ⇒ ceiling 40, start 20.**

Settled knobs for a restart: `wgp_step_max_nm=20` (start), `wgp_cap_max_nm=40`, `wgp_reuse_k=5`, `wgp_reuse_travel_nm=40`, `wgp_fom_slack=1.5e-3`, d1u also `wgp_lam_margin_nm=0.2`.

**Convergence is PREDICTIVE, never "N flat iterates"** (user, 2026-08-31): stop when `dT_pred = ∇T·step` < the 0.002 T noise floor on **3 consecutive accepted iterates**, or the cap is pinned at its 2 nm floor by rejects; fallback 5 accepted iterates with cumulative ΔT < 0.002. Check the reject CAUSE first — noise-corrupted rejects mimic convergence. Lane arbitration = `dT_pred`/hour.

**λ-policy (user ruling).** The λ-hold is an **ALGORITHMIC tool, not a spec**. Slight drift is acceptable and must never cost T; if restoration is ever measured fighting T, **WIDEN `wgp_lam_margin_nm`** (0.05 → 0.2–0.5) rather than fight, and trim the final device by pitch (measured-free, task 49). **W stays the only hard spec.**

### 5.5 How the adjoint gradients are obtained

- **Two adjoints per iterate.** The adjoint source is `dJ/dfield`, built from the objective; `dT/dfield ≠ dW/dfield`, so `∇W` cannot be post-processed out of the transmission adjoint. Adjoint 1 = the **port mode** (`FDTD::ports`, `source port` switched). Adjoint 2 = a **weighted field-region source** whose profile is literally `dsoftW/dI` — sharply peaked **at the two half-max crossings of the envelope** (the width gradient is asking "how do I move the half-max points?").
- **`∇T` and `∇W` come from the SAME solved fields at zero extra cost** — gradient assembly is linear in the objective's Jacobian, so re-running the assembly with a different selector yields a different component out of the same physics. Total = 1 forward + 2 adjoints = **3 solves/iterate**.
- **Tiling (the engineering result that made the width adjoint affordable).** The full-width field-region source was rejected by a per-source **CUDA kernel-launch bound**; splitting it into **4 narrow sources enabled in ONE solve is exact** (sources superpose linearly; the gradient is linear in the adjoint field). MEASURED: **~1.8 h/gradient on GPU vs 8.7–12.1 h on CPU** (single measurement 3133 s vs 8.7–12.1 h). Size ladder (job 136799, 2D rungs, cells at dx 50 nm): 2112×29 FAIL | 2112×14 FAIL | 528×7 PASS. Threshold un-bisected between **3,696 (pass) and 29,568 (fail)** cells; full region = **61,248** cells. ★TRAP: the CUDA error surfaces **~22 min LATE** (the engine meshes on CPU first) — judge a rung on EXIT, never on elapsed time.
- **`C_field`** — the adjoint-field scale/phase calibration. The port adjoint needed `adj_phase_fix=True` with **C = 1.0561 + 0.1239i**: MEASURED over 14 params at TWO operating points the phase is **UNIVERSAL 6.71°/6.67° (Δ0.04°)**; best within-point C gives all 14 signs correct (including the cavity: unfixed adjoint +1.35e-3 vs true FD −3.60e-5) with magnitudes ×0.84–1.67; amplitude varies ×1.5 between points so the campaign C is the geometric mean at the universal phase (worst-case global bias ×1.22). An analogous width-side `C_field = 0.4554 − 0.1336i` (arg ≈ **−16.4°**) was fitted; the literature check calls a *fitted complex* constant a documented bug signature and demands an **invariance test** (fit at ≥2 λ, ≥2 mesh sizes, ≥2 monitor spans, ≥2 device lengths: constant ⇒ legitimate normalisation, drifts with mesh ⇒ missing dV, with λ/position ⇒ phase-reference bug, with device size ⇒ tiling/assembly wrong). ★Do **not** fit `C_field` on a CROPPED region (its softW is a different functional). The projection architecture does **not** require `C_field` (direction is scale-invariant); only "exact mode" does.
- **Resonance chain-rule / IFT term (defect #19's fix).** `∇W` from the adjoint is `dW/dp` at FIXED λ, but W is specced at the device's own MOVING resonance:

  `dW/dp = (dW/dp)|_λ + (dW/dλ)·(dλ_pk/dp)`

  `dλ_pk/dp` is obtained by implicit differentiation of the stationarity condition `∂T/∂λ = 0`, from **two extra autograd selector passes over the already-solved fields — ZERO extra adjoint solves**. The estimator is the **MATCHED pair**:

  `gLam = −(g_hi − g_lo) / (T'(λ_hi) − T'(λ_lo))`

  For any lineshape `T = A(p)·S(λ−λ₀(p))` with `S` EVEN, the amplitude part is even and cancels in BOTH antisymmetric differences, so the stencil truncation cancels in the RATIO — **exact for any h, any symmetric lineshape, amplitude drift included**; no curvature is ever formed. Because truncation is gone a **WIDE stencil is BETTER**: at `k = round(0.5·fwhm/dl) = 20` the matched error is **0.0034%** where the NAIVE form (central difference of ∂T/∂p over a second difference of T) is **49.38% LOW**. Closed form for the naive error = **1/(1+x²)**, `x = h/g`, verified to the digit. Guard: `dTp = T'_hi − T'_lo < 0` ⟺ the stencil straddles a maximum; `dTp ≥ 0` ⇒ **LOUD skip**, never a silent revert to the fixed-λ `∇W`. Accepted residual: a **δ-leak of 0.60%** from the argmax index sitting ≤ dl/2 off true λ₀ (removable only by fitting λ₀; not implemented).
  Provenance of `wg_dwdlam = 0.3655 µm/nm`: `gates/derive_dwdlam.py` reproduces **0.3654 (0.03% from stored)**, but ONLY with the filter rule stated: unique (λ, W) pairs AND `fom > 0.5·max(fom)`. The slope is **NOT universal** — 0.3655 (uniform, r 0.984, n=9) vs 0.2958/0.300 (seesaw, r 0.849–0.867) = a ~20% spread; re-derive for a new seed family. `wg_dwdlam_fit=True` refits it online from the run's own accepted (λ,W) points (n≥5, span ≥0.5 nm, ±30%/refit clamp, abs [0.10, 0.70]).
  MEASURED validation: λ-detrending the two cancelled baselines shows **93% (uniform) / 94%-with-the-uniform-slope, honestly 77% (seesaw)** of the width growth that killed them was pure resonance drift. Head-to-head (137873_41 vs _46): the control's `dw_pred` is NEGATIVE every iterate (−0.0002/−0.0162/−0.0221) while measured ΔW is POSITIVE (+0.0110/+0.0122) — the fixed-λ gradient predicts the **wrong direction**; the corrected arm's sign is right. `dlam_pred` +0.045/+0.0505 nm vs measured Δλ_pk +0.040/+0.040 (13–26% error; the old framework predicted 0).
  HONESTY LIMIT: `corr(λ, T) = 0.9963` (uniform) / `0.9965` (seesaw) — T and λ are nearly collinear in that data, so "T gain at fixed λ" is **not separable** from those logs. **Do NOT quote a "13× better exchange rate."**

### 5.6 The best designs (MEASURED) and where the vectors live

All 191-vectors live in `runners/lumopt2_design/best_designs.py` — **import them, never re-paste**. All numbers below are **PVA design numerics**.

| design | job(s) | t_pk | λ_pk (nm) | W = fwhm_env (µm) | Q_L | Q_i | loss |
|---|---|---|---|---|---|---|---|
| **`BEST_D1_T9676`** (d1 lane, BEST-seeded) | 139225 → 139520 | **0.96762** | 1566.4440 | 18.2901 | 2019.8 | **123 737** | 0.03158 |
| d1u (uniform-seeded, a DIFFERENT basin) | 139226 | **0.96341** | 1565.8141 | 18.5445 | 2010.2 | 108 850 | 0.03542 |
| **`BEST_T9636`** (hand/optimizer benchmark) | 136465 eval 12, CONVERGED | 0.96361 | 1566.444 | 18.3531 | 2021.6 | ~110 000 | 0.03639 |

- **d1 = +0.00401 T over the benchmark** at a slightly NARROWER mode and the identical resonance — the first machine-driven improvement over `BEST_T9636` in the programme's history. λ held **exactly** (0.0000 nm residual) on every d1 iterate; d1u drifted only under the biggest caps (0.22 nm).
- d1u came within **0.0002** of the benchmark from a uniform seed via a different family: mean corrugation ~321 nm (sub-uniform, innermost teeth ~266) vs BEST's ~357.95 nm (super-uniform) ⇒ **two distinct basins exist.**
- `BEST_T9636` geometry: mean corrugation **357.95 nm**, cavity width **961.1 nm**, cavity elongation `2·Σshift` = **132.6 nm**, winner comb. Conformal re-measure of the same device: **T 0.97805**, λ 1560.907 nm, FWHM 19.008 µm, Q_L 1714.2, **Q_i 155 358**. Cavity loss `1−T` fell 0.0717 (uniform origin) → 0.0220 = **−69%** while the mode width was KEPT (−0.88% vs origin).
- Counts: d1 = 17 evals / 13 iterates / 1 reject; d1u = 40 evals / 15 iterates / 1 reject + 4 width trips. Every branch (reject → cap halve → retry from stored gradients, restoration, recenter, WidthTrip, optstate resume across a crash) executed on hardware for the first time in that generation.
- Supporting measurements: reuse smoke **job 139345 task 52 COMPLETED exit 0** (`[proj 1]`/`[proj 3]` logged `width row REUSED — width adjoint skipped`, both held constraints, cap grew). Angle probe **job 139256 task 53**: the width gradient rotates **0.685° per 10 nm of travel** (cos 0.999929; vectors in `results_from_athena/v2_ns2_toy/gW_angle_{A,B}.npy`).
- Local artefacts: `results_from_athena/d1_generation/` (all logs, optstate sidecars, base fsp), `results_from_athena/v2_ns2_toy/`, `results_from_athena/lumopt2_v2_proj_c1/`, `results_from_igum/lumopt2_v2_proj_b1/`, `results_from_athena/v2_gpu_gradient_pause/jsonl/` (78 jsonl, 721 kB), `results_from_athena/fsp_exports/`.
- Published per-tooth-profile page: https://claude.ai/code/artifact/468acf70-db1e-4c70-8516-042876384cf4

Other MEASURED facts about the design levers (keep):

- Tooth shifts buy **transmission, not linewidth**: scaling BEST's shifts ×0/×0.5/×1.0/×1.5 gives T 0.93613 / 0.95222 / 0.9635 / 0.96747 while Q stays flat 2078 / 2078 / ~2020 / 1977. **~0.938 is the no-shift ceiling**; the gap to 0.9636 is the shifts' value (+0.025 T). Apodization SATURATES (see-saw amplitude peaks at d=90, T 0.93836; d120 0.93634, d150 0.92677).
- The project's Q (λ / spectral FWHM) barely discriminates: **1930–2109 (~9%)** across designs spanning T 0.901–0.964. Judge by Q and you conclude backwards.
- Decomposition at the origin width: depth-only −0.029, +cavity +0.041, +shifts +0.053, +shape +0.005 (total +0.069) ⇒ "just a deeper grating" REFUTED.
- The comb is worth **+0.0040 T at benchmark width** (136465 iter0 comb-ON 0.9630@18.279 vs 136491 comb-OFF 0.95877@18.3449), +0.0030 at 17.853, +0.0107 on the uniform origin — only ~2× the 0.002 floor, so "small but real", never "significant". The 57-post comb is now a **fabrication decision**, not a physics necessity (kept in, user's call). Separately, at N=165 with a clean control the comb is the ONLY lever measured to give a large gain at constant width: ΔFWHM **−0.35%**, Q_i **+17.1%**.
- The 46 nm corrugation step at the tooth-25/26 free/frozen boundary is **NOT the mechanism** (de-step 136302 t24: +0.0006). Outer free teeth 20–25 are inert.
- Beating Itai's device at a common operating point needs **Q_i ≥ ~610k (his N=98) / ≥1.16M (N=130)**, i.e. T ≈ 0.9966 at Q_L 2000. TM cannot get there (TE/TM Q_i factor **3.4×** measured on his own geometry; light-cone headroom 10.0% vs 5.5%). Realistic TM ambition: **Q_i 150–250k**. Beating him needs a **TE lane** running this machinery.

### 5.7 Every bug found and fixed, with its signature

1. **`profile_line` y-row-0 bug** — signature: every `sigma_um`/FWHM logged before 2026-08-18 is VOID; widths ~10× under-reported growth; σ ratio 1.013–1.015 while true width ran +19%. Fix: replicate `extract_and_process_field_profile` exactly; profile fetched ONCE/eval; raw `(x,|E|²)` saved to `<out>/profiles/<label>_evNNNN.npz` (~30 kB) so all future metric questions are answerable offline with zero GPU.
2. **Defect #19 — the gradient sampled at a STALE λ.** (a) `make_func` pinned the single-λ twin `field_profile_adj::wavelength center` to `spec._wg_lam_track`, a CONSTANT to autograd (zero Jacobian row, zero dEps) ⇒ `dλ_pk/dp` structurally absent; (b) a ONE-EVAL LAG — `_wg_lam_track` was set in the log callback AFTER the eval, so eval N used eval N−1's resonance (eval 0 fell back to `scan_center_nm`). Signature: `softw_um` vs `softw_adj_um` gap grew 6× in one iterate (+0.0010 → +0.0061); measured ΔW fully explained by `0.3655·Δλ` with nothing left over; the exchange rate never improved. Fix = the IFT chain term of §5.5.
3. **Naive λ-stencil, 49.4% low** — caught by `gate_lam_chain.py` before any GPU time. Fix: the matched pair.
4. **Flat-`x` selector IndexError** — `self.fct = lambda x: anp.abs(x[0])[i_lo]` killed **job 137267 at 2:03:18** on `IndexError: invalid index to scalar variable`. ★THE FACT: the fct's `x` is the **FLAT** vector `[T(λ_0) … T(λ_{n_wl−1}), softW]`, not a list of FOM entry results — which is why `x[-1]` is the width. Correct form `lambda x: anp.abs(x[i])`. Root cause: **the math was gated (0.0034%), the CALL PATH never was.**
5. **Band-edge negative-index wrap** — with `i_pk ≤ 1` or `≥ len(wl)−2` the k-clamp still produced `i_lo = −1` and the stash read `T[i_lo−1]`, so **numpy negative indexing WRAPPED to the far end of the spectrum** and silently built `gλ` from the wrong band edge. Guard `1 < i_pk < len(wl) − 2`; swept all 501 peak positions → 0 out-of-range reads.
6. **Peak-RAM doubling** — stashing `gfields_Tlo/Thi` alongside `gT` + `gfields_W` would hold FOUR field sets; the double pass alone already OOM-killed a 160G job at 501 λ (**job 137012, exit 137**). Fix: run the selector passes BEFORE the width stash and convert each to a 191-float parameter vector immediately (`gvec_Tlo`/`gvec_Thi`), `del f` before the next is built ⇒ peak stays at TWO live field sets. (Needs `spec._wg_p = p` stashed before `project.compute_gradient(p)`; `_grad_from` became dead and was deleted.)
7. **Noise-slack ratchet** — the filter tested `fom > acc.fom − slack` while `acc` was overwritten on every accept, so each step could lose up to the slack and **the reference walked down with it**. Signature: d1's last 4 accepted iterates drifted DOWN 0.71832 → 0.71647 fom (−0.0021 t_pk) at slack 1.5e-3. Fix: `fom_ref = max(acc["fom"], fom_best)` with `fom_best` updated AFTER the filter test.
8. **Reuse staleness is ANGULAR, not per-iterate** — `wgp_reuse_k=5` was justified from 0.685° **per 10 nm of travel**; d1's cap grew 25→38→57→60 nm while reusing 3 deep ⇒ ~180 nm stale ≈ 12°, far past the ~2.8° assumed. Fix: `wgp_reuse_travel_nm = 40` (reuse only while travel-since-fresh + next cap ≤ budget; 4 reuses at cap 10, at most 1 at cap 60; travel accumulates post-clip and persists in the sidecar). ★General lesson: **when a knob is validated at one operating scale, re-derive it in the units the physics uses before combining it with a knob that changes that scale.**
9. **ns2 width-trip response was the penalty-era one** — under the projection the width is steered by the STEP, so a trip is a step-size failure; the legacy handler ratcheted `corr_max_nm` 451→429→407 nm, fighting the optimizer. Signature: d1u churned 4 trips (W 18.99 / 19.05 / 19.11 µm = +3.5…+4.1%, outside ±2%), zero progress for hours. Fix: under `wgp_ns2` a trip **halves the persisted cap** and forces a fresh width row; `corr_max_nm` untouched.
10. **`lams.ptp()` numpy-2 crash** — the ndarray method was removed in NumPy 2.0; the Athena container is numpy 2.x while IGUM is 1.x, which is why the IGUM b1 lane ran the same code for days. **d1u job 139050 FAILED at 11:39 h**; d1 139049 was one eval from the same crash. The branch (`wg_dwdlam` refit) engages only at **n ≥ 5 accepted points**, and every gate/toy/smoke ran ≤4. Fix: `np.ptp(lams)` ×2 (~lines 2801/2810). ★Lesson class: count-triggered branches need a gate case AT the threshold.
11. **Trust-clamp silently did nothing when the seed sat ON a bound** — old form `min(r, p0−lo, hi−p0)` plus a symmetry requirement collapsed `r_eff` to 0 exactly when the clamp matters most (uniform seed's shifts = 0.0 = the lower bound). MEASURED consequence (136709 ev2): L-BFGS-B's unit-norm scaled probe ≈0.0575/param over a 100 nm half-range = 5.75 nm/tooth = elongation 287.4 nm, width 32.27 µm — **13.6 µm out of band**, one wasted 90-min eval, the same 0→287 lurch that crippled campaigns 136466/136640. Fix (2026-08-24): clamp ASYMMETRICALLY against the box; the trust region is a search-step control, NOT a tightening of `shift_bounds` (which stay 0–200).
12. **Rank-deficient fwhm_wall** — see §5.4 reason (c). Interim fix shipped (default-off): `FW_TOOTH_W` 3-block per-tooth slopes in µm/(nm·tooth): inner-8 **−2.925e-3**, middle-9 **−2.384e-3**, outer-8 **−0.268e-3**; `8(−2.925e-3)+9(−2.384e-3)+8(−0.268e-3) = −0.047000 = FW_A_MCORR` exactly. Superseded by the projection architecture.
13. **The width cheat: `Σshift` reconstructs the excluded cavity-LENGTH knob.** MEASURED (seedB eval, job 54488): T 0.9585 via σ 19.176 µm (**+9.6%**, far outside the deadband) with ρ 0.9888 fully compliant; all 25 shifts +5.1 nm mean, cavity +9.9 nm, corr/avg/comb ~0. Mechanism: `2Σs` = +255 nm ≈ half a period ⇒ λ +2.6 nm ⇒ mirror penetration up ⇒ mode widens. Fixes: `BETA_ELONG = 1e-5/nm²` with deadband `|2Σshift| ≤ 120 nm` (violator scores 0.182 vs its 0.015 illegitimate gain), and `_best_from_log(…, sigma0_um)` now selects the best width-COMPLIANT row (it previously had NO width filter, so a WidthTrip restart resumed AT the violator = a ~1.5 h/cycle burn loop).
14. **Double-wrapped guard exceptions** — lumopt2 wraps fct exceptions (`scipy_optimizer.py:583` without `from e`, `optimization.py:852` with), so `except RecenterNeeded` never matched and **IGUM job 54309 died** instead of recentre-restarting. Fix: catch `RuntimeError` too and unwrap **both** `__cause__` and `__context__`; genuine errors re-raise.
15. **Label reuse = silent wrong resume.** `run_campaign` cold-start-resumes via `_best_from_log` on `<out_dir>/<label>_evals.jsonl`. Task 41's label `lumopt2_v2_proj_toy` was **the same label the CONTROL (137075_41) wrote under**, so the corrected run would have started at the control's iterate-2 point (fom 0.669780, W 18.3684) instead of the uniform seed — destroying the comparison and burning ~9 GPU-h. Jobs 137267 and 137296 both carried the flaw (neither reached an iterate). Fix: bump the label per attempt (`lumopt2_v2_projchain_toy`).
16. **My own h5 cleaner killed a healthy campaign.** `h5_clean_once.sh` pass 1 (`-mmin +30`, keep newest 2 `*_output.h5`) deleted the FORWARD h5 mid-gradient on a slow resumed iterate ⇒ two resume incarnations died identically at their first gradient with `Can not find result 'E' in field_profile_adj`. Fix: `-mmin +240`, keep newest 4. Earlier the same cleaner **could not free space at all** (kept newest 2 per study dir while each `*_files` dir holds only 2–3 files) — quota hit 289G/300G with 101.2 GB in 77 h5 files; 84.6 GB freed by hand (→204G). ★Do NOT diagnose a cron janitor with `pgrep`.
17. **Mesh-phase artifact** — `fwhm_env` at dx=50 nm mis-reads up to **3.9%** (design-dependent standing-wave sampling phase). Fix: campaigns migrated to `eng.DX_PITCHLOCK_NM = PITCH/10`; the dx=50 campaigns 136141/136188 were cancelled because their width channel was an artifact.
18. **Dead adjoint source** — an "import source at z = 0" width adjoint produced ALL-ZERO fields (z=0 is the mirror plane with anti-symmetric BC; fwd twin plane `Ex = Ey = 0.0` exactly), so the 2026-08-23 "GPU width-adjoint proven / 52 min" claim is **VOID** (source-less solve). Guard `check_import_src_injects` now raises on a dead source; `gates/h5_gate.py` reports `max|E|` per component per file so a dead source is visible immediately. Keep-forever FD reference vector: **[−0.00365, +0.01825, +0.02026]** for [corr_1, shift_1, wcav].
19. **Tooth-gradient miscalibration (the parked era).** MEASURED α = adjoint/FD at a detuned point, both mesh refinements: corr_1 5.6/5.1 | corr_25 3.9/7.7 | shift_1 14/16.4 | comb r 1.02/1.29 | comb x 1.30/1.32 | comb d sign-flip. Adjoint **×5–16 TOO LARGE** on teeth, ×29 on cavity (an earlier "×5–16 low" was a tuple-order misread — `validate_gradient` returns `(fd, adj, err%)`). dEps (CAD side) PROVEN CORRECT locally (volume-integrated |dEps| vs analytic 1.10 corr_1, 1.01 comb r) ⇒ the deficit is in the **field-contraction** stage. lumopt2 dev246 has **no dielectric-boundary correction** anywhere. `bc_patch` (Johnson E∥/D⊥) measured **≤0.04%** and `colocate_fields` **~1e-6** ⇒ neither explains it; mechanism partly OPEN. Resolved in practice by the C-fix of §5.5.
20. **Config-override trap** — `cfg.geometry.corrugation_depth_m` is correct; `cfg.grating.corrugation_depth_m` silently creates a dead attribute and the device builds at the default. Also caught by gate B1: cavity width comes from the GLOBAL avg 800 nm via `cavity_width_option="avg"`.
21. **`Box(dx=dy=dz=50e-9)` wiring** and **`KeyError('T')`** (port expansion results carry only `"S"`; `T = |S21|²`) — both fixed in the B-gate era. **`project_folder`** had to be set explicitly or all sim files landed in the container's ephemeral overlay and vanished (which is what "port expansion missing" actually meant in job 132624).
22. **Invented deploy flags** — `--no-submit` produced a stray 10-task array **133070**; `--no-dispatch` produced a duplicate campaign driver **54440** an hour later. The real flag is **`--upload-only`**. Both deploy scripts now ABORT on unknown flags.
23. **`validate_gradient` compared a penalty-WRAPPED adjoint against a RAW FD** — gate 136189's shift Re "−0.0352" IS the elongation-penalty gradient, not physics. Engine gates fixed to raw-vs-raw. Related asymmetry: the deployed α prints put the κ-penalty in the adjoint but not in the FD.
24. **Task-index reachability** — indices 27 AND 34 were eaten by `_GFR_RUNGS(27-36)`; `predispatch_check.py` now audits reachability programmatically.

### 5.8 The local gates — run these before ANY dispatch

All zero-GPU, all in `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\gates\`.

```bash
python runners/lumopt2_design/gates/run_all_gates.py          # runs the four below, fail-fast
python runners/lumopt2_design/gates/gate_projection_local.py  # expect "ALL PASS" (15/15 + section 9)
PYTHONIOENCODING=utf-8 python runners/lumopt2_design/gates/gate_lam_chain.py
python runners/lumopt2_design/gates/gate_lam_chain_plumbing.py
python runners/lumopt2_design/gates/predispatch_check.py      # "ALL SEEDS IN BOUNDS"
```

- **`gate_projection_local.py`** — drives the REAL engine (`_proj_step`, `_ns2_step`, `make_fct_v2`, `CampaignSpec`) with synthetic gradients; no lumapi, no FDTD. Checks null-space exactness (`gW·d == 0` to <1e-12, worst measured 8.3e-17), the scaled-space round trip, clip, restoration, legacy-off, and **section 9** = the three 2026-09-01 fixes including a **must-fail teeth check** proving the old acc-anchored filter accepts the whole downhill sequence.
- **`gate_lam_chain.py`** — the λ-stencil math against an analytic Lorentzian with a KNOWN `dλ_pk/dp = 0.037` **and a drifting amplitude** (`DADP = 0.02`) so a pure translation cannot make a wrong estimator look right. Asserts MATCHED ≪ NAIVE.
- **`gate_lam_chain_plumbing.py`** — builds the REAL `make_fct_v2` over the REAL flat layout and runs `autograd.jacobian` on each selector, asserting a **one-hot** jacobian AND that the **old broken form still raises IndexError** (a gate that cannot fail proves nothing).
- **`predispatch_check.py`** — reproduces EXACTLY what the runner does to the vector (seed → `detune1` → `BEST_T9636` seeding → clamp) and checks it against `param_bounds(spec)`; also audits task-index reachability. This class of bug cost FOUR dispatches in one night.
- **`derive_dwdlam.py`** — provenance of `wg_dwdlam = 0.3655` (reproduces 0.3654); documents the in-band filter rule.
- **`h5_gate.py`** — runs on the **Athena login node** (`python3 h5_gate.py gfr_full gfr_yhalf gfr_quart`); reports `max|E|` per component per file so a dead adjoint source is caught immediately.
- Not a local gate but mandatory per CLAUDE.md §5: any change to lumopt2/adjoint/driver code runs the **PIPELINE SMOKE** first — `validate_c325` **task 47** (projected lanes) or **task 50** (ns2 lanes): same 191-param spec and code paths on an N=60 low-Q surrogate, ~1.5–2 h; its numbers are never quoted as physics. (An old pointer to "task 35" is STALE — that is a GFR CUDA-probe rung.)
- Also: `python debug_fsp_compare/scene_snapshot.py --out <tmp>` diffed against `debug_fsp_compare/snapshots/` (6 configs) after any `bragg_device.py` geometry/monitor edit.

### 5.9 What is undeployed or owed, and the ranked residual risks

**Cluster is IDLE** (jobs 139520 + 139226 CANCELLED 2026-09-01 after their state was fetched; server-side result dirs and optstate sidecars intact for resume).

**Undeployed / uncommitted** (all local, all gated, `compileall` + gates green):

- the three 2026-09-01 engine fixes (slack anchored to `fom_best`, `wgp_reuse_travel_nm`, ns2 width-trip response), all **default-inert**;
- `gate_projection_local.py` section 9;
- `BEST_D1_T9676` in `best_designs.py`;
- `HANDOFF_2026-09-01.md`, HANDOFF.md's top box, and the CLAUDE.md / skill / memory rule updates.
- Deliberately NOT deployed while lanes ran: swapping optimizer policy risks a REQUEUE picking it up mid-campaign. **Deploy is bundled with the restart.** Committed baseline: `af40902` ("inv design working version 1") and `744b4f1` (audit fixes), both pushed. **Commit needs user approval.**

**Exact resume recipe** (from the repo root):

```bash
bash athena/deploy_athena.sh --upload-only          # 0. push the fixed engine, no dispatch
python runners/lumopt2_design/gates/run_all_gates.py   # 1. gates green first
SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
  bash athena/deploy_athena.sh \
  --lumopt2-design=runners.lumopt2_design.campaign_v2_proj_d1     # or ..._d1u
```

- **`4d_1g` REJECTS 300G** (275G QOS cap, the sbatch error is buried) — use **256G**.
- The optstate sidecars currently hold `cap_nm` **60.0 (d1)** and **25.3 (d1u)**. **Reset d1's to 20.0 before restart** (edit `lumopt2_v2_proj_d1_optstate.json` on Athena: `cap_nm` → 20.0, `reuse_age` → 0, `reuse_travel` → 0.0, `reuse_W0` → null) or it resumes at the cap that broke it. Local copies of both sidecars are in `results_from_athena/d1_generation/`.
- **Never re-derive across labels**: a campaign continuing a toy/lane copies `<label>_evals.jsonl` + `_optstate.json` into the new label server-side before dispatch. Never dispatch a seed/benchmark re-measure — cite the stored row. The only legitimate seed forward is the one inside an optimizer iterate whose FIELDS feed the adjoint.

**Next steps, in priority order (all undispatched):**

1. Restart both lanes with the §5.4 knobs (cap ceiling 40, travel budget on). ~1.5 h/iterate, ~35% less on reuse iterates.
2. **`N_FREE` 25 → 60 — the highest-leverage lever left.** Itai's apodization spans 60 periods/side; we free only the innermost 25 and the converged profiles drop abruptly exactly at tooth 25. The adjoint gives all parameters' gradients from the SAME three solves, so this is nearly free per iterate (~296 params; only the ~6 min dEps assembly scales). Needs: `N_FREE` widening, a bounds/predispatch pass, one N=60-free smoke.
3. **Free the comb** (115 params currently frozen at slivers; historically +17.1% Q_i, width-neutral) as a separate arm so attribution stays clean.
4. **TE lane** — the only route to contesting Itai's absolute numbers.
5. Parked method upgrades: adaptive `k` from the live `gW_refresh_cos` telemetry; Feppon's separate null/range step caps; a mode-identity (profile-overlap) check per iterate to catch mode hopping; damped null-space L-BFGS if a lane stalls with everything else healthy. **Adjoint parallelisation was evaluated and REJECTED** (≤6% left once reuse is on, plus a queue wait and a licence seat per refresh).
6. Also parked: the commit; the code-compaction consolidation (the `_rgp_step` surgery landed, the rest is unstarted); deleting `scratch_s5vec.txt`.

**Ranked residual risks:**

1. **O(Q²) ill-conditioning** (arXiv:2511.16643 §2.1): the dominant Hessian eigenvalue of a fixed-frequency resonant objective scales as Q². We DO re-centre on the measured peak; we do **NOT** have an explicit λ-drift trust region / bandwidth constraint — the cheapest literature-endorsed hardening available. Step sizes tuned at low Q will not transfer.
2. **The fitted complex `C_field` (arg −16.4°) is unfalsified** — the invariance test (≥2 λ, ≥2 mesh, ≥2 spans, ≥2 device lengths) has not been run. Only "exact mode" depends on it.
3. **Fano / vanishing-curvature failure of the IFT term**: for a Fano profile the T maximum sits at detuning 1/q, not at the QNM frequency, and `∂T/∂λ = 0` can have a second root a stencil could hop to; `dλ_pk/dp = −(∂²T/∂λ∂p)/(∂²T/∂λ²)` is unbounded as `∂²T/∂λ² → 0`. A guard on `|∂²T/∂λ²|` is **missing** (only the `dTp < 0` sign guard exists).
4. **`wg_dwdlam` carries ~20% magnitude uncertainty across design families** — mitigated because it cancels entirely from ns2 feasible directions, but it still enters diagnostics and any non-ns2 lane.
5. **Width noise floor never measured** — the metric's box-to-box floor is 0.03%, but the per-eval scatter of the live read has not been measured; ΔW comparisons at ~0.01 µm may sit inside it.
6. **δ-leak 0.60%** in `gλ` (argmax index vs true λ₀), accepted.
7. Only the surrogate N=100 PVA lane has been optimised; the **production confirm** (N ≈ 165–169, accurate mesh, conformal) is the only reportable truth gate and has not been run on d1. The user's ruling: do NOT conformal-re-measure now.
8. The tooth-gradient contraction mechanism (§5.7 #19) is only **empirically** corrected by C; a Lumerical version bump could double-correct (standing order: diff `fom/port_fom.py::_compute_adjoint_fields_phased` — the `1j·ω/4·conj(am)/P` line — and re-run the 2-sim quadrature calibration).

### 5.10 The standing bans

- **CMT width models: BANNED** (user: "delete all cmt use"). The coupled-mode-theory width model was deleted everywhere; it had been "validated" against the void widths, and the user's physics objection stands independently (tooth-scale moves violate the slowly-varying assumption). ★Scope note: CMT **is** authorized for the separate q3db predictive-engine programme; the ban is scoped to the lumopt2 optimizer / width-wall.
- **σ (second moment) and participation-ratio metrics: BANNED FOREVER.** σ errs up to 24 pp, PR 21 pp — all L²/moment widths are blind to core flattening.
- **The raw-line FWHM metric: DELETED** (`fwhm_raw_of_line` / `mode_fwhm_um`), together with the fitted `FWHM_A_RHO = −74.3 µm/unit` / `FWHM_A_SHIFT = +0.01948 µm/nm` slopes and the FWHM/σ shape alarm's 0.978 reference.
- **Width is measured ONE way only**: `sim_helpers.extract_and_process_field_profile` == `fwhm_env` == `post_processing.fwhm_m`. Do not reintroduce any variant.
- **The 2-pillar pair is permanently dropped** ("pillar pair no more"); "pillars" means the periodic row.
- Cavity LENGTH as a free knob was excluded by the user ("pure λ-tuner, it will just confuse us") — the `Σshift` cheat of §5.7 #13 is the optimizer reconstructing it.
- LDOS / Q-V objectives REJECTED (conflate Q and V). Solo comb optimization REJECTED (solved analytically by the comb rules). Comb-only campaign REJECTED by user. Binary/count comb params NOT in v2. Dip seed OUT (+2.5% at birth > band).
- **lumopt v1 is a source of LESSONS ONLY** — never a runtime component, no reverting, no v1-oracle runs.

### 5.11 The other four optimization families, and why each is or is not used

1. **lumopt2 (the live path)** — ships with Lumerical 2026 R1.2+ at `/opt/lumerical/v261/api/python/lumopt2` (Athena container) and `~/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/api/python/lumopt2` (IGUM); version string `0.0.1.dev239+g5d890d897` (R1.2) / `0.0.1.dev246+g14ebc81f2` (R1.3), `__version__ == '0.0.0'` at runtime (setuptools-scm placeholder) so R1.2 and R1.3 lumopt2 are **indistinguishable at runtime**. FOM = an arbitrary autograd function of monitor values; only TWO monitor metrics exist (PortResults `'transmission'`, FieldResults `'intensity'` = Σ|E|²) — everything else `NotImplementedError`. `dEps/dp` is **finite-difference THROUGH THE MESHER** (auto dp ≈ range·4.9e-4 ≈ 0.1 nm — too small vs dx=50; use explicit dp ≈ 1 nm, comb 2–5 nm); 2 mesh/index evaluations per parameter per iteration, no extra FDTD solves but linear in n_params. **Exactly ONE optimizer class ships: `ScipyOptimizer`** (L-BFGS-B default; `max_line_search` default 8). **NO resume** (ours is bolted on via `_best_from_log` off the eval jsonl). `SlurmRunner` is broken as shipped in both R1.2 and R1.3 (imports the nonexistent `lumopt2.utils.lumslurm`; fix = `import lumslurm; sys.modules["lumopt2.utils.lumslurm"] = lumslurm`, auto-applied in `lumopt2_design.py::import_lumopt2`), also marks jobs done without verifying success and `run_dependencies` launches only the first dependency. `LocalRunner(resource="GPU", max_retries=2)` batches fwd+adj into ONE `runjobs`; license failures surface as LAYOUT_MODE and auto-retry ×2. FieldResults is **single-λ only**. R1.3 added FDTD symmetric/anti-symmetric **boundary-condition** support in the adjoint pipeline (NOT a mirror-geometry parametrization — mirroring must be encoded in the `func`). No fabrication constraints anywhere. `visualize_geometry`/`visualize_fom` block on `input()` — never call on a cluster. Docs: https://lumerical.docs.pyansys.com.
2. **lumopt v1 — the Ansys fork** (`api/python/lumopt`, bundled alongside lumopt2, zero cross-references, no deprecation notice). **NOT** classic chriskeraly lumopt: breaking 2-tuple FOM API (`get_fom → (fom, fom_wavelength)`, `fom_gradient_wavelength_integral → (grad, grad_vs_wl)`), a new `porttransmission` FOM (© 2025 Lumerical, hard-codes port names 'fom'/'source', lacks `adjoint_source_name`), FAID beta (fabrication-aware, `enable_FAID_beta`), one_forward co-optimization, soft-min (log-sum-exp) multi-FOM. **Historically we fixed its adjoint ourselves**: a 4-fix stack took `vec_error` **11.40 → 0.144** (79×) — (i) `target_T_fwd_weights` propagation patch keeping `w(λ)` explicit, (ii) `frequency dependent profile = 1` on both ports, (iii) `multi_freq_src=True`, (iv) an empirical **0.5× kernel factor**; plus `mesh_override_dxyz_nm = 25`, `use_concurrent_adjoint_solves = False`, and `scale_initial_gradient_to = 0.25` (the default 0 made the first L-BFGS-B step ~0.034 nm = sub-Ångström, far below the mesh, so the Wolfe search rejected everything and the optimizer exited after 1 iteration; a companion bug returned params in scaled [0,1] space and the post-opt verification simulated `cavity_width = 299` instead of 798.6). Residual at 0.144: `cavity_width` ratio 1.52. **Status: lessons only, never run.** Value kept: `lumopt/utilities/gradients.py::boundary_perturbation_integrand` as the correct E∥/D⊥ boundary-integral reference.
3. **Gradient-free Lumerical PSO — `runners/gradient_free_design/`** — a Python/numpy PSO driving per-particle FDTD through lumapi; same FOM (peak T over the bandgap), same incremental-save format, fully working. It is the fallback when gradients are untrusted, and it is also **priced out** for the 191-param problem: ~85 min/eval × 30 particles × 50 iterations ≈ months. The TM parametric-`.fsp` route is DEAD (a parametric .fsp built a dead TM device) — use `rebuild_per_particle`, which is also what filled the 300 GB home quota once.
4. **FD-gradient runner — `runners/fd_gradient_design/`** — scipy L-BFGS-B with a user-supplied central-difference jac on peak T, reusing `gradient_free_design._evaluate_particle` so the cost function is identical to the PSO path. Default start `[300, 300, 0, 0, 800]` via `regular_grating_start`; **11 FDTDs per gradient call** (1 base + 2×5), fewer if the bound guard makes a one-sided difference (e.g. `shift_i = 0`). Deploy `bash athena/deploy_athena.sh --fd-gradient-design=<spec_module>`; smoke `runners/fd_gradient_design/smoke_test.py` (n_periods=20, max_iter=1, ~15 min). Walltime ≈ 11 × ~3 min × max_iter ≈ 33 min/iter at n_periods=80. It existed because v1's adjoint was broken; it does not scale to 191 params (11 → 383 FDTDs per gradient) so it is **superseded**.
5. **Lumerical-native `addsweep('Optimization')` PSO — `runners/lumerical_native_optimization/`: BLOCKED.** In headless lumapi, `setsweep('type','Optimization')` is **silently dropped** — diagnostic job 80331 read back `'type' = 'Values'` with every optimizer property (`optimizer type`, `maximize`, `tolerance`, `maximum generations`, `Run mode`) `None`. Seven attempts (80223 / 80289 / 80300 / 80331 / 80346 / 80358 / 80395) each exposed a different silent rejection; the canonical KB LSF-script-via-`fdtd.eval()` pattern fails with `LumApiError: 'Failed to evaluate code'` (parse error before execution), and `getsweep(...)` introspection fails the same way. Next step if ever revisited is GUI bisection (build the optimization sweep in the Designer, diff the saved .fsp XML) or an Ansys support ticket. **Do not spend more time on it** — the Python PSO covers the need.

---

## 6. Infrastructure facts

### 6.1 Lumerical versions — SETTLED, do not deliberate

**All three environments run 2026 R1.3, build 4572 (FDTD Solver 8.35.4572).** Nothing to configure, choose, or check before a run.

- **Athena:** `~/containers/lumerical-2026R1.sif` — the filename is unchanged **on purpose** (~6 job scripts hardcode it) and now CONTAINS R1.3.
- **IGUM:** `LUM_HOME` in all 6 `igum/jobs/*.sh` → `~/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261`.
- **Local Windows:** `config.py` `LUMAPI_PATH` → `C:\Program Files\Lumerical\v261` (= R1.3). Also present but unused: `v252` = 2025 R2.2 (8.34.4251).
- Cross-cluster lockstep re-proven by canaries **Athena 131295 / IGUM 52223** against the corr-325 N165 anchor — exact. Do not raise the version topic, re-verify the engine, or offer to switch unless the USER asks or a NEW release needs installing.
- History (kept, not options): local R1 (8.35.4413) < Athena R1.1 (8.35.4474) < IGUM R1.2 (8.35.4522) < R1.3 (8.35.4572). No 2026 R2 exists.
- Athena also has a **native** Ansys tree (`/apps/ansys`, `/ansys_inc` → `/usr/local/ansys`, `root:ansys` mode 750; the user was added to group `ansys` gid 3000 in 2026-08-11 so it is readable) — but the only version there is **Ansys 2025 R1 at `/apps/ansys/v251`** (build R251RC2P02). Going native on Athena would be a **downgrade** plus a §2 numerics change. The container wins.
- **2026-09-11 upgrade check — verdict NO on all three:** (a) **Lumerical 2026 R1.4** (notes 2026-09-08) lists only Synopsys/OptoCompiler workflows, RCWA k-vectors, STEP import, MQW threads, viewport speed, a Cloud Burst checkbox and a "don't store source field" option — **nothing** on FDTD GPU, ports/mode expansion, FieldRegion, mesh, lumapi, lumopt2 or licensing ⇒ **stay on R1.3 build 4572**. (b) **PyLumerical `ansys-lumerical-core` 0.4.0** (2026-08-28, Beta) is a pip shim around the installed lumapi; no engine, no lumopt2/lumslurm in the wheel, still GUI-license + save-then-solve — switching is one import line for zero capability gain. (c) **`ansys-lumerical-mcp` 0.1.0 Alpha** stays a workbench layer, on user decision only, never the pipeline. Official lumopt2 still has no resume/cluster runner. ★At each new release read the notes for **THREE triggers only**: lumopt2 gradient/adjoint changes, FieldRegion-on-GPU fixes, a resume or cluster runner. Absent those, don't bump. Ansys notes list features only, so "nothing listed" ≠ "nothing fixed" — a bump still needs the canary.

### 6.2 The two clusters

**Ask which one before dispatching** (a plain one-line question), unless the user already named it.

| | **Athena** (default) | **IGUM** (ECE faculty) |
|---|---|---|
| access | `ssh evyatarrubin@athena.technion.ac.il` / alias `ssh athena`; Technion VPN required (the FQDN does not resolve off-VPN) | `ssh igum` = `evyatarrubin@132.68.58.101` (igum-login1); FQDN does not resolve — use the IP |
| deploy | `bash athena/deploy_athena.sh` → `results_from_athena/` | `bash igum/deploy_igum.sh` → `results_from_igum/`; `REMOTE_BASE ~/research/bragg_sim_igum` |
| Lumerical | **container** `~/containers/lumerical-2026R1.sif` | **native** extracted-RPM tree we own; admins' `/apps/ansys/Lumerical-2026-R1.2` kept as fallback |
| containers | apptainer | **IMPOSSIBLE** — no apptainer/singularity, no module system; `docker-ce` installed and user in the `docker` group but `docker info` is DENIED (verified 2026-08-12) |
| SLURM | `--gpus=1` + QOS; partitions all `MaxTime=UNLIMITED`, the QOS is the limit | `--account=acct-lumerical --partition=part-lumerical --qos=qos-lumerical`, `--gres=gpu:N` form; A100s via `--qos=qos-preempt --partition=part-preempt` |
| preemption | **EVERY GPU partition is `PreemptMode=REQUEUE`**, `a100-public` included (measured 2026-08-14, re-checked 2026-09-11). `PreemptType=preempt/qos` and the **contrib QOS (priority 10000, MaxWall 7d) preys on every lane we have** ⇒ no preempt-proof lane exists. GraceTime 10 min; `JobRequeue=1`, `MaxBatchRequeue=5` | `part-lumerical` `PreemptMode=OFF`; group partitions (part-efrats/-ykasten/-silbmark/-ugproj) `PreemptMode=OFF`, MaxTime UNLIMITED; `part-preempt` IS preemptible (requeue) |
| limits | QOS `24h_1g`: **100 submitted / 4 running** per user | `qos-lumerical`: MaxWall **7 days (10080 min)**, NO per-user job/submit cap (bound = 7 GPUs), Priority 10000. `qos-preempt`: MaxWall 7 d, MaxJobsPU 16, MaxSubmitJobsPU 40, ≤16 GPUs. `MaxArraySize=1001`; SLURM 23.11.4 ⇒ up to 7 guaranteed + 16 preemptible = 23 concurrent |
| hardware | default partition list `h200-shared,a100-public,rtx6k-shared,l40s-public,l40s-shared` (fastest memory first). Since 2026-09 **`a100-public` = 5 nodes / 40 A100** (n305, n307, n308, n310, n313 — the migrated DGX hosts); pin with `--gpu=a100` when queue wait matters | `part-lumerical` = ece-alecohen1 (3× RTX A4500) + ece-alecohen2 (4× RTX 2080Ti), 16 CPU / 230 GB each. `part-preempt` = 48× A100 (ece-efrats[2-5], ece-silbmark[1-2]) + 8× RTX PRO 6000. A4500 ≈ 1/3 A100 ≈ 1/7 H200 for FDTD; 20/11 GB VRAM ⇒ big-domain runs stay on Athena |
| known weakness | login node **kills all user processes at ssh logout** (nohup and tmux both die); site lua `cli_filter` **FORBIDS `sbatch --wrap`** (script files only) | **slurmdbd is DOWN** (`sacctmgr`/`sacct` → connection refused localhost:6819) — use `squeue` + `scontrol show assoc_mgr` + file mtimes; login sshd flaps; hand-maintained Lumerical tree. Headless via `QT_QPA_PLATFORM=offscreen` (no Xvfb); plain `fdtd-engine` only (`-ompi-lcl` broken: no libmpi.so.40); `fdtd-engine -v` failing on `libglut.so.3` means `scilibs` is not on `LD_LIBRARY_PATH`, NOT a broken install |

- **Division of labour (measured):** both clusters earn their keep via PARALLEL throughput (two seeds in one night = the convergence evidence). IGUM's **compute** is fine but its **infrastructure** is the weak link ⇒ give IGUM long self-contained resume-protected runs; keep interactive / closely-monitored / fast-iterating work on Athena. Long single solves (>≈3 h) prefer IGUM ("empirically calm", not proven immune).
- **Login-node connection budget: ≤~3–6 ssh/hour per cluster, ONE connection per poll** (fold lmstat/log/queue probes into the same ssh). IGUM began refusing the key ~80 min after a monitor polled it 24×/h; ~45 min of zero contact restored it. On any auth refusal with port 22 open: STOP all automated contact ≥45 min, then ONE probe — never retry-loop. Cluster JOBS are unaffected by login-node auth.
- **ssh/scp form:** always host-first, `ssh evyatarrubin@athena.technion.ac.il "..."`. Never env-var-prefixed forms — they evade the permission-rule pattern matching (including the `scancel` ask-guard). Strip the banner with `grep -vE "post-quantum|openssh|may need to be upgraded"`. Use the plain command form (no `cd &&`, no pipes) — compound forms are blocked by the permission classifier.
- **Cluster scripts are a maintained PAIR: `athena/` + `igum/`.** Any edit to `athena/scripts/*` or `athena/jobs/*` is mirrored to `igum/` in the same change or explicitly reported as not mirrored (a 2026-07-11 audit found real drift).
- IGUM-specific trap: **simultaneous Lumerical STARTUPS on one node race the per-user `ansyscl` daemon and die at 60 s** — `Could not open 'fdtd': appOpen error: ... did not produce the startup UUID within 60 s` + `ANSYSLI exited or could not read server port ansyscl.<node>.<node>_<user>_261`. This is **NOT** seat starvation (it happened with 31/50 seats free). MEASURED three hits in one night: 62750 (`%4`, 1 of 4 died), **63415 (`%2` onto a busy node: 4 of 4 died within 17 s — a cascade)**, while 63202_0 and 63195_3 started alone minutes apart and were fine. ⇒ on IGUM dispatch anything that opens a lumapi session with **`--max-concurrent=1`**; recover dead indices with a staggered `--array-tasks=<lo>-<hi>` after the queue drains.

### 6.3 The license server

- One FlexLM server for everything: **`ANSYSLMD_LICENSE_FILE=1055@132.68.48.51`** (lmgrd) and **`ANSYSLI_SERVERS=2325@132.68.48.51`** (vendor/ANSYSLI). The deploy scripts export both, so jobs check out **by IP**. This is the SAME license the user's PC uses; **seats are SHARED between Athena and IGUM** and with the rest of the faculty.
- Features: `lum_fdtd_solve` **50 issued**, `lum_fdtd_gui` 50 issued. No `lum_fdtd_engine`, no separate Accelerator/HPC feature.
- **★Seat check is MANDATORY before any dispatch of more than one task, and REACHABILITY ≠ AVAILABILITY.** Open ports only prove the server answers. Probe the COUNT **from IGUM** (Athena's lmstat is the false negative): `$LUM/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve`. The pool oscillated **39–46/50 within hours**. Budget concurrency vs FREE seats (array task ≈ 1 seat, lumopt2 iteration ≈ 2); bands: **≥35/50 in use = HIGH (hold fan-outs), ≥45/50 = CRITICAL (no new dispatches)**. `LocalRunner`'s 2 auto-retries are blip cover, not a plan. The ~6-concurrent-solve ceiling is an UPPER BOUND, not a guarantee — our queues being empty proves nothing.
- **The two starvation signatures (measured 2026-08-04):**
  1. **IGUM (native) = loud instant death** — bare `in run:` + "Unable to checkout the requested HPC license" (that day: "requires 12 licenses for feature FDTD_Solutions_engine"). Cheap: deaths are instant.
  2. **Athena (container) = SILENT no-op** — `fdtd.run()` returns in **~1 s** with no fields, and the pipeline later crashes with `Can not find result 'expansion for port monitor'`. That error has **TWO** causes — shared-`.h5` clobber OR a license no-op — **disambiguate via the log's `Simulation time`: ~1 s = license, normal solve time = clobber.** A third cause: the sim genuinely never ran, or its files landed in the container's EPHEMERAL overlay (write sim outputs only under bind mounts).
- **Canary-first rule:** opening a second cluster or resuming after ANY license anomaly = 1 task first, fleet only after it logs a real solve time.
- **★Athena `lmstat -96` is a FALSE NEGATIVE — do NOT block a dispatch on it.** `--license-probe` / container `lmutil lmstat -a -c 1055@132.68.48.51` returns `-96` ("lmgrd is not running / server down"; locally `WinSock: HOST_NOT_FOUND`) even when the license works: lmstat status-enumerates by the server's advertised FQDN `lumerical-lm.ece.technion.ac.il`, which does not resolve from Athena nodes — a path real checkouts never use. Reliable signal instead: TCP **1055 and 2325 OPEN by IP**. (2026-06-30: preflight said "down"; job **115369** then ran real 7-min solves.) On IGUM lmstat IS reliable (the FQDN resolves there).
- A **genuine** outage looks different: ports 1055/2325 **connection-refused** from both clusters and IGUM lmstat gives `-15,570 Cannot connect to license server system` (measured 2026-07-28; recovered 2026-07-29, <1 day).
- **Multi-GPU per single sim is BLOCKED at the license tier, project-wide.** `setresource("FDTD", 1, "processes", N>1)` returns no error but the readback stays `'1'`. `addresource("FDTD")` adds parallel CAPACITY, not acceleration: 1 vs 2 resources gave identical wall clock (161.9 s vs 161.6 s). Multi-process FDTD needs an Accelerator entitlement the Technion seats lack. `N_GPUS` was removed from `athena.conf`; `--gpus=1` is hardcoded at the four sbatch sites. IGUM is identical (same license) and additionally has **no mpirun/mpiexec anywhere**. ⇒ throughput parallelism only, via job arrays (`--option3`). The new Athena QOS `24h_16g` (2 nodes / 16 GPU, MaxJobsPU 1) and IB multi-node are therefore **useless to us**.
- Legacy: on **Zeus** (Technion PBS, Lumerical 2021R2.5) a user `~/.config/Lumerical/License.ini` with `domain=1` silently breaks lumapi (`appOpen error: Failed to start messaging, check licenses...`) because it overrides the system `.ini`'s `domain=2` + floating server. Fix already in `zeus/jobs/run_python_job.sh` and `zeus/jobs/run_fsp_job.sh`: export `ANSYSLMD_LICENSE_FILE=1055@132.68.48.51` and `ANSYSLI_SERVERS=2325@132.68.48.51` before launching, which overrides the .ini files. Launching the GUI on the Zeus head node re-creates the bad .ini (interactive runs break; submitted jobs stay protected).

### 6.4 The container, and how it gets updated

Use the **`update-container` skill**. The 5 GB `.sif` **never crosses the VPN**, and **no old version's artifacts are ever deleted** (user rule): old sifs are renamed `lumerical-2026R1.2.sif`, `lumerical-2026R1.1.sif`, plus the parked `~/lum_r1*_parked_*` trees.

Pipeline (all steps verified working, R1.1→R1.2 then R1.2→R1.3):

1. **Source tree onto Athena.** The LINX64 package is ONE ~1.1 GB RPM (`rpm_install_files/Lumerical-2026R1-3-a4e7f95b355.el8.x86_64.rpm`, md5 `2de375ae217aec6128082cd3c8b66526`). VPN throughput is **not a constant** — measured 10.6 MB/s on the R1.3 push (upload < 2 min) vs 0.19 MB/s earlier; measure before planning an overnight push. Extract on Athena (Rocky 9 has `rpm2cpio`): 12,898 files / 4.0 GB; `v261/VERSION` = 2026R1 / 3 / 4572.
2. **Verify** with an md5 manifest (`find v261 -type f | sort | xargs md5sum` at source, `md5sum -c` on Athena) — 12898/12898 OK.
3. ★**Order matters: LAN-stream the tree to IGUM BEFORE launching the build job** — the build `mv`s the stage into the sandbox. 4.0 GB over the Technion LAN ≈ 5 min. IGUM has **no `rpm2cpio`** (Ubuntu, only `cpio`), hence the extract-on-Athena hop.
4. **Sandbox surgery script** (`~/lum_r13_build.sh`, kept on Athena): `apptainer build --force --sandbox` from the live sif → `mv` the OLD `/opt/lumerical/v261` and `/ansys_inc/v261/licensingclient` OUT to `~/lum_r13_parked_*` (**never deleted**) → `mv` the staged `v261` in → `cp -a` its inner `licensingclient` to `/ansys_inc/v261/licensingclient` (the engine hardcodes that path) → `chmod a+rX` plus the 5 `bin/*` and 3 `licensingclient/linx64/*` executables → `sed` the version in `.singularity.d/labels.json` + `runscript.help` → `APPTAINER_SQUASHFS_COMP=gzip apptainer build --force ~/containers/lumerical-2026R1.sif.new <sandbox>` → in-sif verify (engine `-v`, engine md5 vs manifest, ansyscl present, env intact, LUMAPI_OK).
5. ★**Run it as a SLURM CPU job, NOT on the login node** (the login node kills user processes at logout; `--wrap` is forbidden): `sbatch --job-name=... --time=02:00:00 --cpus-per-task=8 --mem=32G --output=<log> <script.sh>`. R1.3 build = **job 131291, COMPLETED in 8m13s**; in-sif engine md5 `79416b77a68703720521ca7e33986ad1` == manifest; quota peaked 273/300 GB. A loud UCX/InfiniBand `ucs_handle_error` backtrace at MPI_Finalize on athena-post is **COSMETIC**.
6. **Swap deliberately** (never inside the build): queue empty of container jobs → `mv` live sif → `lumerical-2026R1.2.sif` (KEEP) → `mv` `.sif.new` → the live name.
7. **Canary gate** = `runners/metal_mirror/engine_canary.py` (renamed from `r12_canary.py`), 1 task, the comb_q3db corr-325 N165 control row. **PASSED on both clusters at R1.3** (Athena **131295**, IGUM **52223**, 50/50 seats free at dispatch): anchor read from `results_from_athena/comb_q3db/results/result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat` = λ **1559.0010 nm**, T **0.490579** (−3.0929 dB), spectral_fwhm **−0.111913 nm**, Q **13930.5**, mode **19.9702 µm**. Athena R1.3 identical in **every printed digit**; IGUM R1.3 T 0.490578 (Δ 1e-6 = 0.0002%), everything else identical. Solves 9470 s / 9488 s. ⇒ R1.1 ≡ R1.2 ≡ R1.3 on this device. **Canary cost: ~2 h 40 m GPU per cluster (~5.4 GPU-h total)** — ~175 µm of grating × 8.0 × 8.8 µm box at dx 50 nm ≈ 50 M cells, and Q ≈ 14k needs ~16 photon lifetimes. Free physics check from the `*_p0.log` at ~2% complete: observed ring-down τ ≈ 13 ps vs Q/ω = 11.5 ps. Auto-shutoff fires near 5.5% complete, so the log's "Max time remaining: 44 hrs" is the no-shutoff worst case, **not** a warning.
- Gotchas: a stale IGUM hostkey in Athena's `known_hosts` after IGUM's key rotation (verify the fingerprint out-of-band, then `ssh-keygen -R 132.68.58.101` on Athena); surgery needs ~19 GB of quota headroom. An engine bump is a §2 **named-numerics change** and ends with a canary vs a stored control on **both** clusters.
- **SLURM commands INSIDE the container** (proven: probe job **132630** sbatch'd from inside, COMPLETED on a compute node):

```
apptainer exec \
  --bind /opt/slurm --bind /etc/slurm --bind /run/munge \
  --bind ~/slurm_env/passwd:/etc/passwd --bind ~/slurm_env/group:/etc/group \
  --bind $HOME/scilibs:/scilibs \
  ~/containers/lumerical-2026R1.sif bash -c '
    export PATH=/opt/slurm/24.11.3/bin:$PATH
    export LD_LIBRARY_PATH=/scilibs:$LD_LIBRARY_PATH
    sbatch <script>'
```

  `~/scilibs/` also holds `liblua-5.4.so`, `libmunge.so.2*`, `libjson-c.so.5*` (deps of the site's `cli_filter_lua` + `serializer_json` plugins — file-binds into `/usr/lib64` do NOT work, a bound DIR on `LD_LIBRARY_PATH` is the reliable pattern). `~/slurm_env/passwd|group` = the container's own files + `getent passwd slurm` appended; ★binding the HOST `/etc/passwd` **breaks LDAP users** (apptainer injects the login user; the host file lacks LDAP entries). lumslurm configs: Athena `~/.lumslurm.config` → `~/slurm_env/fdtd-engine-container.sh` and `python-container.sh` (host-side wrappers that re-enter the container with `--nv` + the scilibs bind), pythonpath `/opt/lumerical/v261/api/python`; IGUM's points at the R1.3 tree. Campaign DEFAULT remains LocalRunner inside one GPU allocation.

### 6.5 Home quota, and the container-init hang

- Home is `hpc-nfs1:/home`: **soft 300 GB, hard 330 GB**.
- **Symptom of being over quota:** jobs sit at `INFO: Setting --writable-tmpfs (required by nvidia-container-cli)` for 10–20+ min with the GPU idle and Lumerical/Python never starting, on EVERY node/GPU type. Looks like a cluster outage. Diagnose with `ssh ... quota -s` — `NNNg*` (asterisk) = over soft quota, writes block, the container overlay setup hangs.
- Cause (2026-06-22): the TM PSO's `rebuild_per_particle` saves a layout `.fsp` + result `.mat` + temp `.h5` per particle; ~130 sims pushed home to 319 GB (`.h5` alone 58.8 GB / 80 files; `.mat` 104 GB; 426 `.fsp`). Fix: delete disposable `.h5` scratch (**never** the `.mat` results) — `find ~/bragg_sim_athena/results -name '*.h5' -delete` freed 59 GB. Always deploy with `KEEP_H5=0` (the default).
- lumopt2 **never** cleans solver scratch (~4 GB per completed study, ~15–20 GB steady state per live campaign). The rolling cleaner is `~/h5_clean_once.sh` on the Athena login node, run from `crontab` (`*/10 * * * *`); its retention rule must be `-mmin +240` / keep newest 4 (see §5.7 #16 — an age floor shorter than the slowest iterate deletes a live forward). Note a cron janitor is invisible to `pgrep`.
- **rtx6k-shared nodes (n317/n318) hang at the SAME step even under quota** — a separate issue: driver 595.45.04 / CUDA 13.2 incompatible with the container. Stick to a100/l40s (driver 570).

### 6.6 QOS table and caps (Athena, MEASURED via `sacctmgr show qos` / `scontrol`)

| QOS | walltime | MaxTRESPerJob | MaxJobsPerUser | notes |
|---|---|---|---|---|
| `24h_1g` | 23:30 | `cpu=32, gres/gpu=1, mem=275G` | **4 running / 100 submitted** | the default `ARRAY_QOS` in `athena/athena.conf` (~line 16) |
| `4d_1g` | 4 days | `cpu=32, gres/gpu=1, mem=275G` | 8 | the lane for long lumopt2 drivers |
| `2h_2g` | 2 h | — | 3 | |
| `12h_4g` | 12 h | — | 3 | used for toys/gates |
| `24h_4g` | 24 h | — | 3 | |
| `72h_8g` | 72 h | — | 1 | |
| `4h_0g` | 4 h | — | — | CPU lane |
| `contrib` | 7 days | — | — | priority 10000; **preys on every lane we have** |
| `24h_16g` | 24 h | 2 nodes / 16 GPU | 1 | irrelevant (1 GPU/sim license cap) |

- `SBATCH_MEM=300G` is **REJECTED** at submit (`sbatch: error: QOSMaxMemoryPerJob` / `Batch job submission failed: Job violates accounting/QOS policy`), and the deploy prints only `ERROR: sbatch failed.` unless you read the full output. **Practical ceiling 256G**; 160G is the long-standing known-good for ordinary port runs. Nodes have 1–2.3 TB physically — the QOS is the limit.
- ★**`ARRAY_TIME` as an env override is SILENTLY IGNORED** (`athena.conf` plain-assigns it after the env is set: passed `24:00:00`, job got `23:30`). **`SBATCH_MEM` DOES work** (`${SBATCH_MEM:-}`). Campaign dispatch uses dedicated knobs `LUMOPT2_QOS` / `LUMOPT2_TIME`. Verify with `sacct --format=TimeLimit` after submitting.
- Arrays > 100 tasks must be chunked (`--array-tasks=1-100`, then the rest as the queue drains). Count queued tasks with **`squeue -r`** — plain `squeue` collapses a pending array to ONE line and silently undercounts.
- Sweep lists are **PER-STUDY** since 2026-08-15 (`data/sweep_list_<study>.txt`), so one study's deploy can no longer rewrite the list a REQUEUEd task of another study will re-read. Parallel deploys are allowed IFF both studies use per-study lists AND the new deploy touches only its own study's files (verify in rsync's itemized output). Any edit to shared engine/builder code ⇒ serialize. `--after=<jobid>` chains a dispatch behind an in-flight job (afterok).
- **Jobs needing > ~2 h MUST persist progress incrementally and resume from it** — loss budget on preemption ≤ 1 evaluation/solve. An unprotected long job is a DEFECT at dispatch time. The FDTD engine itself has **NO checkpoint/resume** (CLI help checked), so a single solve is ATOMIC and unprotectable by logging; re-check for engine checkpointing at every version bump.
- Stopping runs is confirm-first: resolve the specific job ID from `squeue`, state it back, let the `scancel` permission prompt be the confirmation, then re-check `squeue`. Use the `stop-runs` skill; never blanket-`scancel`.

### 6.7 Memory-footprint measurements (host RAM; `sacct` does NOT track GPU VRAM)

RAM is **monitor-driven, not domain-driven** (≈ cells × freq points × 6 components × bytes/complex).

| run type | peak `MaxRSS` | evidence |
|---|---|---|
| monitors-OFF convergence sim, 6× transverse domain | **3,461,724 KB ≈ 3.3 GiB (~3.5 GB)**, `AveRSS == MaxRSS` (flat) | job **95855**, `--mem=256G` test, COMPLETED on an A100-40GB (n310) in 26.5 min — used ~1.4% of the allocation |
| far-field + 2D XY/YZ/XZ fields, TE, pitch 500, 5λ domain, 2001 freq pts | **58,127,324 KB ≈ 55.4 GB** | job **96422**, COMPLETED on an L40S in 25.4 min |
| full 2D/3D field-profile monitors ON | **> 100 GB** | user-reported 2026-06-17 |
| monitors-OFF **ports-only** sweep, converged box 6.8×8.8 µm, **7001** wl points | OOM at default `--mem` | job **116974_2**; the same box at 3001–4001 pts runs fine. Fix used: cut to 4001 pts (50 pm over 200 nm) rather than bump memory — job **116979** |
| lumopt2 canary tasks | 6.5 GB (vs 160G requested) | — |
| SweepSpec pipeline | peaked 68 GB | — |
| lumopt2 multi-entry FOM at 501 λ | OOM-killed at 160G, **exit 137** | job **137012**; `get_fields_at_wavelengths` fetches the FULL λ grid then slices (`fdtd_session.py:1355`), so every FOM entry pays full-grid — port entries too, even at zero jacobian. Mitigation: 151 λ for gates, 250–256G for field-heavy campaigns |

`sacct -j <id> --format=JobID,State,ReqMem,MaxRSS,AveRSS,Elapsed` — `MaxRSS` lives on the `.batch` step line, not the allocation line. Per-run override without editing shared scripts: `SBATCH_MEM=256G bash athena/deploy_athena.sh ...`. For monitor-light jobs 64G is ~18× overkill; for field-heavy large-domain runs expect to exceed it.

Also: **reduce field data server-side before downloading.** The link runs ~0.5–1 MB/s; a full field-profile `.mat` is ~650 MB/case while a figure needs one plane at one λ (~1 MB). Athena's login-node `python3` has numpy/scipy — extract the slice there. (2026-07-02: 2.5 GB pulled for 4 images before switching.)

### 6.8 The retired DGX fork

- **DGX cluster shut down 2026-09-14; `dgx/` was DELETED from the repo 2026-09-11**, along with `container/nvml_tramp.c`, `container/build_nvml_tramp.sh` and the VS Code DGX task. Its nodes (n305/n307/n308/n310/n313, 8× A100 each) are now Athena **`a100-public`** on current drivers. **Never recreate a third fork** — the forks measurably drifted (2026-07-11 audit: dgx was missing two athena fixes).
- Historical failure mode, kept as history only: DGX ran driver **R470 / CUDA 11.4** while the 2026R1 container expects R5xx+. An `nvml_tramp` shim made GPU detection succeed (`device type readback = 'GPU'`), FDTD printed `Simulation time: 3.452 seconds` and exited with **0 MiB allocated** — the CUDA kernels could never launch and Lumerical did not surface it as an error. Post-processing then crashed on `getresult("FDTD::ports::Port_1", "expansion for port monitor")`. Signature in `~/bragg_sim_gpu/jobs/logs/lum_array-*.out`: a short `Simulation time:` followed by the port-expansion `LumApiError`.

### 6.9 Local verification on Windows — MATLAB and other quirks

- MATLAB R2025b at `C:\Program Files\MATLAB\R2025b\bin\matlab.exe`. Unlike FDTD (which stays on the clusters), `matlab_plotting/*.m` is verified locally:
  - **static lint:** `matlab.exe -batch "msgs=checkcode('matlab_plotting/foo.m','-string'); disp(msgs)"`
  - **headless render smoke test:** build a dialog-free copy inside MATLAB (`fileread` → `strrep` to hardcode result paths, replace `clear; clc;` with `clc; set(0,'DefaultFigureVisible','off');`) → `run(tmpPath)` → `exportgraphics` each figure to PNG → read the PNG to eyeball it.
- Gotchas: MATLAB identifiers cannot start with `_`, so invoke a temp script via `run('path/_tmp.m')`, never bare `_tmp`. **Do NOT use PowerShell `Get-Content`/`Set-Content` to copy or edit `.m` files** — PS 5.1 mangles the UTF-8 `µ` / `—` / `→` characters and MATLAB then throws "Invalid text character"; build temp copies inside MATLAB (`fileread(...,'Encoding','UTF-8')` + `fopen(...,'w','n','UTF-8')`). `-batch` pwd is the launching shell's cwd (repo root), so use absolute paths.
- **All local verification runs are SILENT:** lumapi always `hide=True` (set in `bragg_device`; pass it in ad-hoc scripts too), MATLAB always `-batch`. Nothing opens a window on the user's screen.
- Windows/VPN quirks: `getent` **lies about DNS on Windows** — use `Resolve-DnsName`. `scene_snapshot` needs `PYTHONIOENCODING=utf-8`. `deploy --results-no-fsp` hangs when backgrounded. Athena/IGUM hostnames do not resolve without the Technion VPN; when a monitor reports both clusters unreachable at once, that is a **local** VPN drop (three independent Technion hosts do not fail together) — do not retry-loop or redispatch; an outage costs visibility, not science.
- Local runs are allowed for: building scenes, `save_fsp`, smoke tests, MATLAB plotting and quick non-GPU checks. Local `fdtd.run()` is slow — only do a real local FDTD run if the user explicitly asks.

## 7. Traps and gotchas — the complete catalogue

### Physics and measurement traps

1. **A "resonance" read from the spectrum sits at ~1570 nm in the passband, not at the defect peak** → `max(T)`/`argmax(T)` finds the global transmission maximum, which is the passband, not the π-shift defect notch → always use the stored `resonance_wavelength_nm` or `plot_transmission.m`'s local peak finder; the pitch-redo regression explicitly had to use the stored field, not argmax (project_tm_pitch_redo_1p97.md).
2. **`spectral_fwhm_nm` comes out negative** → `widths*dw` with `dw<0` because the λ axis descends as frequency ascends → use `Q = λ_res / |spectral_fwhm_nm|`; fixed in the bisection driver's `_q` and in `plot_tm_periods_match_te.py` (project_tm_period_match_te.md).
3. **Two different "FWHM" fields exist per .mat and get swapped** → `spectral_fwhm_nm` = spectral width of T(λ) (Q = λ/FWHM); `fwhm_m` = SPATIAL energy-envelope width along x (e.g. 7.67 µm at N=80) → never use `fwhm_m` for a λ-domain weight (reference_spectral_vs_spatial_fwhm.md).
4. **Q_i looks precise but is garbage near T→1** → `Q_i = Q_L/(1−√T)` amplifies T error by `A = √T/(2(1−√T))`: A = 39.3 at T=0.975, 21.7 at 0.95, 9.2 at 0.90, 1.7 at 0.50 → measure Q_i at T ≈ 0.5–0.8 and never quote it from a T>0.95 row without saying so (project_q3db_measurement_method.md).
5. **Above Q_L ≈ 5e4 the standard window silently under-samples the peak** → 20 nm / 4001 pts = 5 pm/sample gives 30 samples across FWHM at Q 10,500 but only 2.2 at Q 143,000 and 1.3 at Q 241,000, so the true peak falls between grid points and T reads LOW → keep 4001 points and narrow the window (3 nm = 0.75 pm, 2 nm = 0.50 pm), keeping ≥1 nm margin for λ drift with N (project_highq_measurement_adequacy.md).
6. **The same high-Q rows are also biased by a truncated ring-down** → `bragg_device.py:769-770` sets simulation time 2000 ps and auto-shutoff 1e-7; reaching 1e-7 needs 16.1·τ, which is 1910 ps at Q 143k and 3219 ps at Q 241k; truncation convolves the Lorentzian with ripples of period λ²/(c·T_sim) = **4.06 pm**, comparable to the linewidth itself → set `TM_SIM_TIME_PS=4000` (~2× runtime) for any rung above Q~1.5e5 (project_highq_measurement_adequacy.md).
7. **A high-Q rung burns its whole walltime and produces nothing** → timesteps ~ ring-down ~ Q, so wall time = `k·N·(t0 + 16.1τ)` with k = 0.00249 min/(N·ps), t0 = 66 ps; against a 23:30 QOS wall **N≥240 CANNOT COMPLETE on Athena** (N=280 was cancelled at 9 h 23 m having needed ~40 h) and Athena's contended shared nodes are **1.67× slower** than IGUM A100s → predict Q, compute the ring-down, and compare before dispatching (project_highq_measurement_adequacy.md).
8. **Requesting `auto shutoff min = 1e-8` makes rows march to the 2000 ps cap / SLURM kill** → the measured total-field energy plateaus near ~5e-8, so the criterion never fires (3 tasks cancelled mid-run) → never request below ~1e-7 (project_autoshutoff_verdict.md).
9. **Relaxing auto-shutoff to 1e-6 "to save time" silently biases Q** → truncation error is a function of **Q only** (five devices, two polarizations, three families collapse on one curve): dQ/Q = −1.7 % at Q 1.28k → −9.9 % at 13.9k → −15.7 % at 26.7k, scaling ~Q^0.7 → production stays 1e-7 for everything; 1e-6 only passes the 2 %Q/0.015 T gate below Q≈1.5k (project_autoshutoff_verdict.md).
10. **Peak T comes back 1.1–1.25 with NEGATIVE loss** → the mode is overflowing the transverse PML: at 200 nm core height, TM n_eff = 1.4585 is only +0.0145 above the cladding, the evanescent tail decays over 1.2 µm and hits the PML at **−10 dB** (want −40) → bump `SPAN_MULT` to 4–5 (−22 dB / −28 dB) plus `SBATCH_MEM`; T>1 is NOT a sim-time/ringdown problem unless Q is in the tens of thousands (project_transverse_domain_size_decision.md, project_tm_wide_mode_corr.md).
11. **The 1.8λ transverse box, validated on well-confined modes, is wrong for TM corr-400 + far-field** → the small box gave T = 0.799 / loss 0.19 (a z-PML artifact) where converged truth is T = 0.886 / loss 0.110 → use `y_span_override=6.8 µm` + `span_multiplier_override=5.42` (z≈8.8 µm) whenever a TM corr-400 far-field run is requested (project_transverse_domain_size_decision.md).
12. **Shrinking y to 4.8 µm makes the device look BETTER** → T rises +0.0090 and radiated fraction FALLS 0.0871→0.0785 as the box shrinks: the near-axial lobe reflects off the y-PML and re-couples, faking +12 % Q_i → a textbook grazing-PML artifact (project_tm_nladder_surrogate.md).
13. **A box that passes the T floor can still be biased** → y=5.8 µm passed |ΔT| < 0.002 but carried Q_i +1.1 % from PML re-injection of exactly the radiation lobe decorations modify, so it does not cancel between designs → judge box convergence on **Q_i, never T**: Q_i is ~11.4× more sensitive to a T error than T is (1/2√T)/(1−√T) at T 0.91 (project_tm_nladder_surrogate.md).
14. **A pre-run prediction that "z cannot shrink at corr-325" was WRONG** → the corr-400 z-requirement was attributed to the TM evanescent tail (a corr-independent mode property) when it is actually RADIATION-driven, so it mostly disappears at corr-325's 8.7 % loss → z converges at 6.8 µm there; z=5.8 is the first bad rung (ΔT −0.0011, Q_i −1.5 %) (project_tm_nladder_surrogate.md).
15. **Absolute T is far more numerics-sensitive than width** → across seven boxes on bare N=100 corr-325, `fwhm_m` spans 19.2411–19.2471 µm (0.03 %) while T_res spans 0.9091–0.9194 (0.010) — T is ~30× more sensitive → never compare absolute T across boxes; always use an in-study control at identical numerics (project_lumopt2_campaign_state.md).
16. **Accurate mesh moves λ_res by ~2.7 nm** → optimization mesh reads λ 1558.57/1558.6, accurate mesh reads 1555.90/1555.95 on the same device (and T 0.80→0.77) → never compare absolutes across mesh modes (project_scatterer_followup_chain.md, project_tm_scatterer_scan.md).
17. **★Two meshers coexist in one project** → `lumopt2_design.py:745` sets `"precise volume average"` while `bragg_device.py:780` sets `"conformal variant 0"`; on the SAME nominal N=100 corr-325 device λ_res = 1564.276 (PVA) vs 1559.006 (conformal) = **+5.27 nm**, and mode FWHM 17.7005 vs 19.2448 µm = **−8.0 %** → never compare a width, λ or absolute T across the two; ratios within one pipeline are fine; PVA ≈ 0.92 × conformal; the ~20 µm spec, the 19.24 µm anchor and the 19.91 µm production value are all conformal (project_mesher_pva_vs_conformal.md).
18. **Which mesher is "right" flipped once** → the engine's own comment claims CT0 staircases grid-aligned tooth edges (so PVA looked better); the 2026-08-21 research digest reversed that — Ansys docs recommend CT0 for high-contrast dielectrics, PVA is documented as a gradient-smoothness tool and is "naive smoothing" (first-order) with a known-sign bias matching our +5.3 nm red-shift → presume CONFORMAL is the better absolute reference; arbitrate with a single-period Bloch-cell dx ladder, not by assertion (project_mesher_pva_vs_conformal.md).
19. **`fwhm_env` mis-reads by up to 3.9 % at dx = 50 nm** → standing-wave sampling-phase artifact, design-dependent → campaigns migrated to `eng.DX_PITCHLOCK_NM = PITCH/10`; the 50 nm campaigns 136141/136188 were cancelled because the width channel was an artifact (project_v2_width_gradient_plan.md).
20. **★A whole q3db ladder sat +1.6 nm above the stored family and nobody could see why** → with `use_z_symmetry=False` the graded z-mesh has no anchor at z=0, so cell count is a rounding knife-edge: port-plane grid 84×**179** (ladder) vs 84×**178** (clean run) — one extra z cell across the 350 nm core moved port n_eff 1.5225→**1.5247** (+0.0022) ⇒ Δλ ≈ +1.67 nm → read the `_p0.log` "Simulation size in gridpoints" line before trusting any absolute Δ from a z-sym-OFF run; fix = `force symmetric z mesh = 1` unconditionally (project_zoff_zmesh_knife_edge.md).
21. **"z-symmetry OFF is clean" does not generalize** → at corr-400/N=80 the change measured null (5 pm), but at corr-325/N=165 the same change gave λ +2.10 nm, T −0.21 dB, Q +7 % → any z-sym-OFF study must carry its own matched control (project_trench_flush_top_study.md).
22. **A stored .mat's own resonance fields can be a finder mis-pick** → the SiO2 hole-lattice rows store resonance 1571.5 nm / T 0.911 / `fwhm_m` 63 µm, which is the PASSBAND; the real collapsed defect peak is 1547.8 nm / T 0.032 → trap for anyone re-reading those files (project_hole_lattice_closed.md).
23. **Same class at N=1300** → stored `resonance_wavelength_nm` = 1490.162 with T = 1.024 (>1, a band edge); the true defect peak is 1492.124 nm with T = 0.0031 → use the defect peak, not the stored scalar (project_tm_h200_w1800_study.md).
24. **At heavy time-truncation the spectrum's argmax is the band-edge lobe** → a 20 ps probe at N=1300 peaked at 1494.9 nm because the defect line is suppressed → do not read argmax as the defect at short T_sim (project_tm_h200_w1800_study.md).
25. **Every stored `*_ff.mat` was projected at the wrong wavelength** → far-field monitors recorded 1 frequency point, i.e. the band-CENTRE frequency (≈1546.4 nm), not the resonance; a stored TE example was 41 % of a linewidth off resonance → new `FarFieldConfig.farfield_freq_points` (default 1 = legacy) records the band and projects at the point nearest `resonance_wavelength_nm` (project_farfield_sph_20um.md).
26. **Far-field "needle at ux 0.96" was the instrument, not the physics** → the side monitor at y = 6.75 µm needs ~33 µm of x travel for grazing rays but the half-span is 30 µm, so the "peak at 0.96" is the clipping edge; the top monitor sees 71 % at |ux|>0.9 and the true aim is the cutoff Λ_c = λ/(n_eff+n_clad) = 530.6 nm → aiming a comb at 536 nm was aiming at a monitor artifact (project_comb_physics_rethink.md).
27. **Moving FF monitors closer contaminates the top monitor** → side at 4.88 µm is fine (2× more horizon power) but TOP at 1.28 µm shows evanescent-tail truncation ripple → keep the top monitor ≥ ~3 µm; T is untouched either way (project_comb_physics_rethink.md).
28. **The default 30 µm far-field monitor x-span clips the corr-400 grazing lobe** (envelope FWHM 15.5 µm, peak at ux = 0.99) → the scatterers program uses 60 µm `FF_X_SPAN_UM` (project_transverse_domain_size_decision.md).
29. **Far-field rows recorded 0.33/0.48 nm off-peak** because λ_res shifts with cavity width W and the monitor λ was fixed → flux underestimated in those rows; shapes still valid (project_tm_loss_new_physics_round.md).
30. **An FFT of the exported mode envelope was wrong by ×1.48 in k** → the Lumerical monitor x-grid is NON-UNIFORM → resample to a uniform grid before any FFT (reference_loss_reduction_options.md).
31. **The numerical noise floor at dx=50 nm is ΔT ≈ 0.0018 and it swallows small effects** → the pillar +0.0020 T sat exactly at the dx=50 jitter floor 0.0018; at dx≈35 nm the jitter collapsed to 0.0001 and the effect survived → measure the floor in-study (repeat points offset by half a mesh cell) then confirm survivors at `simulation_mode="accurate"` (project_tm_scatterer_scan.md, CLAUDE.md §2).
32. **Q values from an under-resolved line are unreportable** → need ≥10 sample points across the spectral linewidth; N190/N215 TE rows were excluded from the Q panel for this reason (project_te_q3db_20um.md, project_target_locking_method.md).
33. **Integer N quantizes the −3 dB crossing** → 1 period ≈ ΔT 0.01–0.02 near T = 0.5, so acceptance T ± 0.03 is a physical floor, not laziness (project_target_locking_method.md).
34. **`κ ∝ corrugation` is solid but `Q_i ∝ corr` is not** → κ_corr325 = 0.0353 µm⁻¹ vs κ_corr400 = 0.0440 (ratio 1.246 vs corr ratio 1.231, 1.2 % agreement) yet Q_i ∝ corr^−1.8 was a MIXED-N fit; the radiative law at fixed N is corr^**−2.90** → separate the coherent (κ) law from the radiative (Q_i) law (project_tm_nladder_surrogate.md, project_q3db_predictive_engine.md).
35. **Extrapolating ln T vs N misses the crossing by +191 %** → dlnT/dN was −0.0058/period at corr 233 but −0.0426/period near the crossing at corr 250, and the corr-233 slope did not transfer → extrapolate `Q_c = Q_L/√T` (linear in N by construction) and never ln T (project_q3db_measurement_method.md, project_te_q3db_20um.md).
36. **"Measure Q_i cheaply at low N and multiply by 0.293" is unsafe** → Q_i drifted 58k→76k (+31 %) over N 110→165 at corr 276 while te_q3db's Q_i was flat within 4 % over N 166–215; and the containment threshold does NOT transfer between devices — Itai's Nt60 TE device sat past the saturation benchmark (end-field 5.5e-3, 2.0e-3) and Q_i still went 610k→1,159k (+90 %) → demonstrate saturation with two measured Q_i values at different N (project_q3db_measurement_method.md).
37. **`Q(−3 dB) = 0.29289·Q_i` is exact algebra but only reproduces measurement to ~2 %** → validated on four directly-measured anchors (TE N166 −1.9 %, TM N165 −2.1 %, TM N170 trench +0.5 %, TM N169 +3.2 %) → a 17 h crossing run buys ~2 % over a 3 h row at T≈0.88; spend it only when the absolute is the product (project_q3db_measurement_method.md).
38. **Lumerical's FDE "TE polarization fraction" labels are INVERTED for this device** → under the x=thickness rotation the fraction is exactly |Ex|²/(|Ex|²+|Ey|²), so device TM (E vertical, Ex-dominant) gets TE-fraction ~0.99 → classify by transverse E power (device TM ⇔ Σ|Ex|² > Σ|Ey|²); `runners/tm/calibrate_neff.py::_pick_mode` uses the inverted rule, so its `neff_tm_avg`/`neff_te_avg` are likely swapped (project_fde_te_tm_label_inversion.md).
39. **"TE barely radiates" was false at current anchors** → TE control at pitch 500 / corr 300 / n 1.97 / N=80 measured T 0.8733 / R 0.005 / **loss 0.1217** — the belief came from the older n_core era and was never re-measured (project_scatterer_greens_program.md).
40. **A 50 nm FDTD-vs-FDE λ offset at h=200 nm is a dz artifact** → `dz = core_height/7 = 28.6 nm` is hardcoded in the bragg_device mesh box, so the FDTD-grid n_eff reads 1.476 vs FDE 1.535 and the grid Bragg wavelength lands ~1488 nm, outside the planned window → a fab/EME comparison at thin cores needs a dz-convergence check (project_tm_h200_w1800_study.md).
41. **The 1D TMM mis-centred a scale ladder** → predicted scale 0.78 → ~20 µm, reality 0.72 → 17.61 µm → do not use it to centre a ladder; measure one row and use the local slope (−1.98 µm per 0.1 of scale in the 20 µm region) (project_itai_hh_apodization.md).
42. **Mode width at a short surrogate is truncation-biased** → bare corr-325 mode FWHM = 16.80/17.74/18.39/19.24/19.66 µm at N = 60/70/80/100/120 = 84/89/92/96/98 % of the ~19.87 µm asymptote; second-moment truncation is worse (44 % of ∫x²I beyond the device end at N=80, 29 % at N=100, 6 % at N=165) → constrain the RATIO σ/σ_ctrl at the same N, never an absolute 20 µm (project_tm_nladder_surrogate.md, project_acoustic_detector_width_spec.md).
43. **★σ (second moment) is essentially BLIND to apodization** → corrugation apodization alone moved FWHM +4.89 % while σ moved +0.001 % (17.2518→17.2520 µm); the live campaigns ran at σ ratio 1.013–1.015 while true FWHM ran +19 % → a 2nd moment cannot see a flattening core whose tails compensate; σ under-reports total growth ~10× (project_lumopt2_campaign_state.md).
44. **Every L²/moment width has the same blindness** → softW tracks measured `fwhm_env` to ≤1.6–2.2 pp across +4.9 %→+26.6 % true growth, while σ errs up to 24 pp and the participation ratio (∫I)²/∫I² errs 21 pp → all moment widths excluded permanently (project_v2_width_gradient_plan.md).
45. **`fwhm_m` is box-independent but crop-sensitive** → 19.2411–19.2471 µm across seven boxes (0.03 %), yet the floor-relative FWHM is strongly crop-sensitive BELOW ~45 µm of x span → cropping ≥51.68 µm changes nothing, cropping tighter does (project_lumopt2_campaign_state.md).
46. **Exchange rates measured at different operating points reverse the ordering** → "elongation is 4.7× better than corrugation" compared a uniform-seed ladder against a retrim curve on a DIFFERENT (apodized, e = 130.6) device; measured on ONE device the ordering REVERSES → never compare exchange rates across operating points (project_v2_width_gradient_plan.md).
47. **★Extrapolating a secant past its measured range cost 2.0–2.7 µm of prediction error** → the out385 single-point rate (8 teeth × 60 nm → 0.1285 µm) was applied at 17 teeth × 90–130 nm, ~6× more amplitude; predicted W 18.638/18.456 vs MEASURED 16.6207/15.7648 → every rate must carry the amplitude range it was measured over; use outside that range is a prediction to test, never a number to design on (project_v2_width_gradient_plan.md).
48. **Asymmetric apodization between the two half-gratings breaks `√T = Q_tot/Q_c`** → the loss reading is contaminated by back-reflection imbalance → keep the parametrization mirror-symmetric or audit with R too (project_inverse_design_cost_function.md).
49. **A "tooth shift" is THREE perturbations, not one** → phase + duty cycle + DC index (`bragg_device.py:226-229` shortens the NARROW gap); receipt: the shift ladder measured λ +1.6 nm per +374 nm of 2Σs, and pure phase redistribution cannot move λ; the DC-index term scales as 1/(n_eff−n_clad), ~2× larger for TM → the TE/TM shift comparison is CONFOUNDED (project_tm_radiation_design_rules.md).
50. **A negative shift is NOT "shortening the wide part"** → narrow-target: narrow HP−s, period 2HP−s, cavity +2Σs; wide-target: wide HP−s, same period, same cavity; negative: narrow HP+s, period 2HP**+**s, cavity **−2Σs** → only narrow-vs-wide holds s, period and cavity fixed, so it is the only clean duty-cycle comparison (project_shift_target_sign_test.md).
51. **A registered prediction for in-core air was an order of magnitude optimistic** (said T 0.83–0.86, measured **0.3813**) → in-core damage was scaled by the Born dEps ratio alone (air/oxide = 1.60), which is invalid in-core: the TM normal-E discontinuity boosts a low-index inclusion by (n_core/n_hole)² = 3.88 vs 1.86 (another ×2.08), the mode is EXPELLED (non-perturbative), and DC index removal dominates the AC interference term; measured λ-shift ratio air/oxide = 5.09/0.67 = **7.6×**, not 1.6× (project_air_comb_study.md).
52. **The cladding sign-flip/phase-equivalence FAILS in-core** → there dx does not only set the interference phase, it changes WHICH material is removed (holes land on narrow vs wide segments), so the DC term moves with dx and the phase circle is confounded: same geometry, dx 398 → λ −0.67 / T 0.8654 vs dx 132 → λ −3.19 / T 0.5438 (project_air_comb_study.md).
53. **Comb + apodization do not stack** → apod10 + comb T 0.9723 vs apod ctrl 0.9770 = dT **−0.0047**; the pair on apod-10 is worse still (−0.0062); apodization already kills the needle so the comb pays its own emission cost with nothing to cancel → comb = uniform-grating devices only. AMENDED: the comb DOES stack when WIDTH IS HELD (+0.0048 T on the width-constrained inverse-designed device) because free apodization buys its needle-kill by letting the mode spread (project_antineedle_comb_stageP.md).
54. **Modularity sign-inverts under apodization** → the [0,270] pair is −27.4e-3 loss standalone but **+26.2e-3** under apod10; the cavity-width null inverts too; A10+W1000/1050/1100 = 0.9674/0.9597/0.9480, all WORSE than plain A10 0.9767 → combined designs must be CO-OPTIMIZED, not composed (project_tm_loss_new_physics_round.md).
55. **Judging designs by Q leads to the opposite conclusion from T** → project Q (= λ/spectral FWHM) spans only 1930–2109 (~9 %) across designs spanning T 0.901–0.964, because Q is set by the shared mirror coupling; scaling the best design's shifts gives T 0.93613→0.96747 while Q stays flat 2078→1977 → shifts buy TRANSMISSION, not linewidth (project_v2_width_gradient_plan.md).
56. **`plot_transmission.m` reported Q = TE 208 / TM 426 and inverted the TE/TM ordering** → the FWHM search used `find(T > half_max, 1, 'first'/'last')` over the zoom window, whose ENDS sit in the passband, so it grabbed window-edge points; correct local-FWHM Q is TE ≈ 1640 / TM ≈ 793 → fixed in-code (walk outward from the peak to the first local half-max crossing + linear interpolation) (project_matlab_q_factor_bug.md).
57. **"Lossy ⇒ low Q" does not apply here** → cavity Q is MIRROR-COUPLING-limited (set by κ): TE has higher Q (strong sidewall κ) AND higher loss; TM has lower Q AND lower loss (project_matlab_q_factor_bug.md).
58. **Real metal at optimization mesh is an instrument artifact, not physics** → the Al wall measured ΔT −0.0274 because staircased metal at dx = 50 nm ≫ the 8 nm skin depth absorbs ~all intercepted power → real metal needs a 5–10 nm mesh override; never interpret opt-mesh metal as physics (project_scatterer_greens_program.md, stage K).
59. **A second scatterer row at Δy = 200 nm broke linear superposition** → 160 nm-diameter pillars at 200 nm centre-to-centre = ~40 nm surface gap = a coupled DIMER; superposition errors 9.5 %/16.0 %/14.3 % vs a 5 % gate → flag any row spacing < ~2r + 200 nm; solve-time constraint became Euclidean centre distance ≥ 270 nm (project_scatterer_greens_program.md).
60. **Constant-α CMT structurally cannot fit this program's Q_i** → constant-α loss makes Q_i independent of L, which cannot represent the measured envelope-limited Q_i ~ N^3 with saturation; any such fit HAD to half-work (right lineshape, wrong loss scaling) → radiation gets its own per-family saturating law injected into CMT, never fitted inside it (project_q3db_predictive_engine.md).
61. **Pure power-law fits through the Q_i knee give garbage exponents** → p = 0.73 artifact vs true ~3.2 + saturation → use `1/Q_i = 1/(A·N^p) + 1/Q_sat` (project_q3db_predictive_engine.md).
62. **The k-space comb model's optimism GROWS with post count** → +0.004 at 31 posts, +0.008 at 62, +0.018 at 122 → rank only, never extrapolate to bigger arrays; its Born amplitude extrapolation is absurd out of range (+0.10 at d = 1.0 µm, f = 59 unphysical) (project_comb_physics_rethink.md).
63. **The light-cone leak ranker is a ranker, not a predictor** → log-log corr 0.975 with slope 0.32 (compressive) against the measured Q_i ladder, and it kept falling over N 165–195 while measured Q_i was flat ⇒ the Q_i saturation ceiling is NOT light-cone-limited (project_v2_width_gradient_plan.md, project_q3db_predictive_engine.md).
64. **"Needle = r⁴ parasitic" was a self-correction** → the Λ-536 phase circle is a clean sinusoid on a PEDESTAL (mean dγ/γ +0.178, swing ±0.25); the model's coherent beam is only 0.047 of it, the rest (0.117) is posts re-scattering the counter-propagating carrier with NO grating phase (m=0), spilled into the cone by the comb's sharp ENDS; its phase is set by the comb's CENTRE x_c, rotating at β−k_c = 0.257 rad/µm (sign flips every 12 µm), NOT by dx (project_comb_physics_rethink.md).
65. **Mode length saturates, so L was never a lever** → N=100→120 grows the mode only 2.2 % (19.245→19.661 µm) while T falls 0.9104→0.8441; corr-325's mirror-limited asymptote is ~19.7–20 µm (project_tm_radiation_design_rules.md).
66. **Duty cycle is not a radiation lever for a first-order grating** → no harmonic reaches the light cone, so suppressing the 2nd harmonic buys nothing (project_tm_radiation_design_rules.md).
67. **`∫T dλ` is sign-inverted for this deliverable** → `∫T dλ = (π/2)·λ·T_peak/Q`, so maximizing band area at fixed peak literally minimizes Q; a Q term would not fix it either, only re-price the broadening → use a windowed high-p soft-max, and scale the window to the MEASURED FWHM each iteration (a fixed-nm window lets broadening leak back) (project_inverse_design_cost_function.md).
68. **A +2 % width increase alone inflates T by ~+0.007 with zero physics change** at the corr-325 N=100 surrogate (κ↓2 % → Q_c↓~7.5 % → T↑) — half the total expected real gain (~0.015) → keep the upper width band tight; the honest ranker under width drift is Q_i = Q_L/(1−√T) (project_inverse_design_cost_function.md).
69. **A "narrowing" effect can be a Q-vs-width trade, not a win** → cavity hourglass/barrel sit on the SAME curve in both directions (hour150 15.370 µm/T 0.8583 … barr150 15.622/0.9069); only the air trench (−0.85 %/+0.0157 T) and cavity W1250 (−0.29 %/+0.0178) fall OFF the curve → and cavity width is NON-monotonic (W1050 +0.59 %, W1150 +0.18 %, W1250 −0.29 %, W1400 −0.74 %) (project_tm_width_reducing_levers.md).
70. **A notch that "cuts loss" can just be riding the width Pareto** → innermost-tooth notch gave −0.0048 loss but at +2.8 % wider mode and UNCHANGED Q (1306 ≈ 1304) → slot removes SiN → wider mode → less radiation, not special recycling (project_innermost_tooth_recycling_theory.md).
71. **An intermediate `⟨x²⟩` / "edge-split −19 %" shape result was a PHYSICS ERROR** → the radiated amplitude was evaluated at unshifted kx instead of a~(kx−β); the β-shifted calculation gives <0.2 % for all in-plane shapes at fixed area (project_innermost_tooth_recycling_theory.md).
72. **The counterdiabatic "loss cut without width cost" was falsified by its own control** → at MATCHED total shift, (fwhm_µm, loss) = lumped-2 (15.47, 0.053) → CD-shape (15.51, 0.050) → uniform-14 (15.99, 0.040): a trivial uniform distributed shift dominates CD on loss; earlier "−31 % / supersedes stack" was OVER-STATED (s3 uniform points mislabeled in a first quick-look) (project_bic_kerker_batch1_dispatch.md).
73. **A width-derived "optimal profile" fakes −11 %** → without an x²-moment guard the derived boundary profile finds the delocalization manifold (a smooth taper); with the guard it collapses to −1.8 % (python_tools/derive_boundary_profile.py) (project_tm_loss_new_physics_round.md).
74. **Moiré is geometrically impossible at 20 µm** → one beat node in a 103.4 µm device ⇒ Δk ≤ 0.061 µm⁻¹ ⇒ mode FWHM ≥ **50.8 µm**; forcing 20 µm needs Δk = 0.393 ⇒ node spacing 16 µm ⇒ 6.5 nodes = a coupled-cavity array (project_tm_radiation_design_rules.md).
75. **x-asymmetry is provably harmful** → A₀ is even, an antisymmetric perturbation gives odd δA, the cross term vanishes, so |A₀+δA|² can only grow; the comb is not a counter-example (it is a separate far-field radiator, not an odd perturbation of the cavity amplitude) (project_tm_radiation_design_rules.md).
76. **A prediction failed because a term was COMMON-MODE** → wide-target shifts were predicted to fall below control (lower ⟨n_eff⟩) but gained, because the cavity absorbs 2Σs whichever segment is shortened, so cavity lengthening dominates T and duty cycle is only the differential → identify which terms are common-mode before predicting a sign (project_shift_target_sign_test.md).
77. **Two spatially separated π-shifts split the resonance into a doublet** → observed; the two-defect/supermode route is de-prioritized and any BIC must be single-resonance (project_bic_scatterer_program.md).
78. **A transmission port cannot harvest a BIC** → a dark mode decoupled from the port is invisible in T, and quasi-BIC reintroduces loss and pulls the bright partner in — exactly the "resonance drains" failure of the 2-cavity FDTD (project_bic_kerker_batch1_dispatch.md).
79. **N=150 T collapsing to 0.184 is NOT a dead device** → deeply undercoupled regime (mirror leak e^{−2κL} ~1e-3 ≪ loss 0.11) makes T hypersensitive to loss, which is the point of the comparison; the dead-device floor is ~0.0008 (project_scatterer_greens_program.md, stage M).
80. **A grazing-needle "retro comb" is optically transparent** → per-row |r| ~1 %, 2-row = 1-row (1.139 vs 1.150), needle bin never reduced (mildly enhanced 1.02–1.15×), and the field-map autopsy shows ZERO standing pattern at the comb row (project_scatterer_greens_program.md, stages H/I).
81. **Blocking ≠ recycling** → the air trench at d = 1200 nm blocks the grazing channel yet measures dT −0.0163 with λ −7.4 nm drag (drain regime); only near d = 1800 nm does blocking convert (+0.0159, needle bin 0.061× ctrl) (project_scatterer_greens_program.md, stage L).
82. **A cos-fit peak did not reproduce** → flat PEC at the fitted optimum d = 2.80 µm gave only +0.0004 vs +0.0019 at d = 3.0 → the 1.6×-floor oscillation was too weak to locate an optimum (project_scatterer_greens_program.md, stage K).
83. **A far-field proxy decouples from T in the last ~1 pp** → monitored-FF/flux ↔ T correlation 0.98 over 97 configs, but trench −23 % FF vs pillars −11 % FF gave nearly equal ΔT; best-FF-predicted combo F (32.4 %) LOSES in T → use FF for placement, T for verdicts (project_inverse_design_cost_function.md, project_scatterer_greens_program.md).
84. **A scalar additive-S21 model over-predicts pairs by ~100 % of the gain** → t0 + Σδt_j predicts singles exactly but gives T 0.9303 vs measured 0.9089 for [0,270]; scalar t has no interference bookkeeping, so two pillars double-count the same recoverable leak → the |b+Σr|² far-field objective (Gram cross-terms) is the correct formalism (project_scatterer_greens_program.md).
85. **An `lsq_linear` solve at 1e-14 scale returned −453 %** → numerical failure at the response-matrix magnitude → normalize the Gram matrix and use L-BFGS-B; verify with a regression row that reproduces a known answer exactly (project_scatterer_greens_program.md).

### Geometry, builder and config traps

86. **★A "no-scatterer control" silently ran a pillar device** → `runners/scatterers/_common.build_ports_base()` returns a config with the scatterer ENABLED at defaults r150/x0/y1000, so a runner that just omits the scatterer fields solves a mirrored pair (job 130913, ~1.75 A100-h wasted; output `..._scR150_X0_Y1000_pair.mat`) → a no-scatterer control REQUIRES `scatterer_radius_nm=[0.0]`; guard = build locally and assert `generate_file_tag(PiShiftBraggFDTD(**cfg.to_device_kwargs()))` equals the stored anchor filename (project_scatterer_default_on_trap.md).
87. **Same trap, different builder** → `tm_scatterer_scan.build_base()` also returns `scatterer.enabled=True` at r=150; `tm_width_lightline` forgot `BASE.scatterer.enabled=False` and job 116970 ran a spurious pillar pair in every row → verify via the stored `scatterer_r_m`/`scatterer_n_sites` in the .mat (project_tm_scatterer_scan.md).
88. **SimulationConfig dataclasses accept UNKNOWN attributes silently** → `cfg.grating.corrugation_depth_m = ...` creates a dead attribute (corrugation lives on `cfg.geometry.*`) and the device builds at the default → after any direct-attribute override verify the BUILT values via `SPEC.expand()`/`describe()`/a build printout (project_lumopt2_campaign_state.md, CLAUDE.md §5).
89. **★A missing module-level `BASE` produced 8 garbage tasks** → spec-mode server does `getattr(module,'BASE',None)` → None → raw default configs: no far-field monitors ("Can not find result 'E' in field_profile"), default small box, 140 s solves, and short colliding tags `layout_N80_avg` that .h5-clobbered 5 of 8 (job 130145) → verify spec runners SERVER-STYLE before dispatch: importlib the module and assert `SPEC.expand(getattr(module,'BASE',None))` rows carry the intended y_span/n_wl/ff; a `__main__` dry run does NOT catch this (project_antineedle_comb_stageP.md).
90. **The transverse box was sized from a SCALAR width and silently lost 448 nm of standoff** → `simulation_config.py:530` sizes y_span from `width_wide_m` (avg + corr/2 = 1184 nm) and never consults the per-tooth arrays whose real max is 1632 nm; same blind spot at `bragg_device.py:821` (mesh-override box) → this is what returned **T+R = 1.045** on Itai's device; set `y_span_um` + `span_mult` explicitly, never derived (project_itai_hh_apodization.md).
91. **The box grew in the wrong direction first** → rung 1 grew only y (3.99→4.44 µm) and got WORSE (T 1.0255→1.0426); rung 2 grew z 3.16→5.03 µm and restored T+R = 0.9576 → z was the culprit, not y (project_itai_hh_apodization.md).
92. **The corrugation convention was assumed wrong twice** → `corrugation_depth = wide − narrow`, so the corr-400 device is narrow 600 / **WIDE 1000** nm with tooth tips at y = 500 (bragg_device lines 861-862 + guard at 0.5·width_wide, line 408); the earlier "tips at 600, y=650 collides" claim was WRONG (project_scatterer_greens_program.md).
93. **The arms are ASYMMETRIC and "x=270" means different things on each side** → cavity length = pitch/2 (spans |x| < ~129); L_wide_1 is LEFT of the cavity, R_narrow_1 is RIGHT (x 129–388, sidewall y = 300), R_wide_1 388–646 (tip 500) → "tangent to teeth at 580" claims were wrong; only ever-contact was the x=0 pillar touching the cavity at y ≤ 480 (project_scatterer_greens_program.md).
94. **A pillar was BURIED inside a tooth without any error** → the round-7 "narrow-touch" row put the right pillar inside `R_wide_1`, so the row was effectively a single pillar; the user caught it from an .fsp screenshot → audit burial explicitly (project_scatterer_greens_program.md).
95. **★The lumopt2 optimizer's device cannot be rebuilt by the SweepSpec path** → `bragg_device.py:1180` right-arm loop uses `s_prev = shift_for_tooth[d-1]` (tooth 1 gets 0) while lumopt2 `make_func`'s right walk uses `s = shift[i]`; a zero-GPU scene diff found **75 mismatching properties, ALL on the right arm, ZERO on the left**, displacing teeth by up to 6.43 nm on BEST_T9636 → NOT repairable by re-indexing (left needs t(d)=s(d), right needs t(d−1)=s(d)); do NOT "fix" bragg_device (it would change every stored distributed-shift result); run the scene diff before taking any optimized vector out of lumopt2 (project_shift_convention_trap.md).
96. **Mesh mode is NOT in `generate_file_tag`** → same-geometry rows at two meshes collide on filenames; the h=200 opt-vs-accurate pair had to become its own module (project_loss_exploration_chain.md, project_tm_h200_w1800_study.md).
97. **File-tag collisions, the running list** → per-tooth-list tags ROUND to whole nm (`ptw80W1002to998` served both δ2.0A and the jitter twin 2.05A — the twin was lost); wide-target shifts needed an appended `"w"` or they overwrite the stored `_S52`/`_S103` narrow rows; `_C{corr}` was added because corrugation is not tagged at W800; `_H{nm}` added for scatterer height; `_avg2W{nm}` added or detuning rows clobber; `_2pishift_Ygap..nm_Xstag..nm_corr2..nm` for two devices; array form `_arr{n}_X{x0}to{x1}_Y…`; `_Ybox{y}_Zbox{z}` only when an override is active; `_smp`/`_ff`/`_fc`/`_dsh`/`_AS{..}` variants; negative scalar tooth shifts are NOT tagged (route through `inner_shift_list`); window centre is NOT tagged (project_tm_superlattice_2L / project_shift_target_sign_test.md / project_target_locking_method.md / project_tm_loss_new_physics_round.md).
98. **Shared output filenames raced and produced empty results** → every config built the SAME `layout_N80_avg.fsp`/`..._output.h5`; concurrent array tasks on the same node clobbered each other mid-run → `getresult("...Port_1","expansion for port monitor")` LumApiError; the same A100 node both passed and failed (intermittent race, not a GPU/port bug) → unique filename per config via `generate_file_tag` (project_side_by_side_coupling.md).
99. **Same-tag rows silently OVERWRITE server copies** → round-10 `[0,270]` and control rows overwrote round-1 files on the server (round-1 was already archived locally); the h4000 trench batch had to be `mv`ed into `results_h4000/` before the full-z rerun (project_scatterer_greens_program.md).
100. **`wall_phase_offset_deg` looks like a κ knob but is GLOBAL-ONLY** → it raises ValueError with apodization/per-tooth/shifts and needs y-symmetry OFF (2× cost); per-tooth taper is an ENGINE change, not a config change (project_tm_radiation_design_rules.md).
101. **Mixed circle + rect scatterers per-site are unsupported** → the narrow-touch + trench combo could not be dispatched without a builder extension (project_scatterer_greens_program.md, stage L2).
102. **Per-site scatterer radii did not exist** → `scatterer_r_list_nm` was UNSUPPORTED (scalar only), so equal-size-post approximations of a sinusoid could not carry the fundamental efficiently; added later as `scatterer_r_list_m` with tag `_scR{min}to{max}` (project_antineedle_comb_stageP.md).
103. **`simulation_mode` changes dx ONLY** → opt 51.7 nm / accurate 36.9 nm, while dy = dz stay pinned ~46–50 nm in ALL modes (`bragg_device` sets dy = 50e-9; device region dy = width_narrow/13 = 46 nm) → the whole scatterer program is effectively optimization-mesh transversely; a gap-rendering question needs a dy knob edit, not an "accurate" run (project_scatterer_greens_program.md).
104. **A single non-mirrored off-axis scatterer with y-symmetry ON silently simulates a PAIR** → `bragg_device` now RAISES on that combination (project_tm_scatterer_scan.md).
105. **Extending added structures into the cladding is pure drain** → the validated pair reproduces +0.0227 at h = 350 nm but measures **−0.0765** at h = 4 µm and −0.0773 at 6 µm (saturated by 4 µm) → pillar recycling is a CORE-LAYER effect (project_scatterer_greens_program.md, stage N).
106. **A Si handle wafer inside the port span hijacks the mode** → "fundamental TM mode" locks onto a high-n_eff Si slab mode → clip the port z-min to Si_top + 0.2 µm; z-symmetry must also be OFF for Si rows and their controls (project_si_substrate_check.md).
107. **`field_profile` had 501 hardcoded frequency points** → ~28 GB at a 1.31 mm grating and `getresult("field_profile")` SEGFAULTED → new inert-by-default knob `cfg.monitors.field_profile_freq_points` (project_tm_h200_w1800_study.md).
108. **Saving a .mat with 101-point 2D field structs hit the MAT-5 4 GiB-per-variable limit** → `MatWriteError Matrix-too-large` after a successful 1 h 48 m solve → cut `N_2D_FREQ_POINTS` 101→41 (~1.6 GB structs, 2.5× margin) (project_scatterer_greens_program.md, stage M).
109. **2D monitors default to the SOURCE limits, not the resonance** → new `monitors.monitor_2d_center_nm`/`span_nm` + `apply_monitor_overrides` branch were needed so per-row 2D windows sit on the MEASURED resonance (project_scatterer_greens_program.md, stage M).
110. **`apply_monitor_overrides` reset the new far-field monitors back to 1 frequency point** → smoke job 164883 FAIL; also `record_2d_fields` produced a 197 MB .mat at N=10 → fixed, and 2D fields turned OFF in that study (project_farfield_sph_20um.md).
111. **★`TM_SIM_TIME_PS` is not reachable from the sweep path** → `athena/deploy_athena.sh:987` (single-run) forwards it in its sbatch `--export` but line 1256 (the `--option3` array path) does NOT, and its only hook `EXTRA_EXPORT` is reserved for `LOCKED_LAMBDA_FILE` → set `os.environ["TM_SIM_TIME_PS"]` at the TOP of the study runner module (imported on the node before the scene is built) and mirror any export change to `igum/` (project_highq_measurement_adequacy.md, project_q3db_measurement_method.md).
112. **`n_wl_points` is fixed at 3001 on the sweep path** (not sweepable) → size the window from the expected linewidth instead (reference_itai_npy_analysis_recipe.md).
113. **`shift_bounds_nm = (0, 200)` looks tightenable and must not be tightened** → pitch is 500 nm so half_pitch = 250 and 200 leaves a 50 nm minimum narrow tooth; an earlier assumption of pitch ≈ 325 nm would have made (0,200) geometrically invalid (project_grating_geometry_facts.md).
114. **`STUDY_DIR_NAME` must be height- AND pitch-aware** → `_evaluate` stamps cache files `result_corrmatch_tm_C<pm>.mat` by CORRUGATION only (no height, no pitch), so a same-height retarget (1550 nm pitch 531.5 vs 1590 nm pitch 546) would reuse stale cache → dirs became `tm_wide_mode_H{h}_P{round(pitch)}` (project_tm_wide_mode_corr.md).
115. **`it11_card_builder.py` reads a Windows-only CSV at import time** → default path `C:\Users\evyat\MATLAB\...\device_names.csv` does not exist on Athena/DGX, so every cards-mode task failed inside `athena_run_one.py` before any sim (job 79349) → sibling copy at `runners/experiment_comparison/device_names.csv`, kept in sync by hand (project_it11_card_csv_deployed.md).
116. **A runner that reads from `results_from_igum/` dies at IMPORT time on the cluster** → the deploy syncs `runners/` ONLY (job 56027, FileNotFoundError) → embed the vector in `best_designs.py` and import it; guard = a local test that imports all runners with `builtins.open` blocked on `results_from_*/` paths (project_v2_width_gradient_plan.md).
117. **Evaluating an evolved 191-vector under a `bare`/frozen spec dies in 34 s on sliver bounds** → bare specs pin comb slots to seed ±1e-3 nm, so evolved radii like 80.0093 are rejected (111 bound violations in a negative test) → use the blessed adapter `replay_params(spec, p)`, which resets inert comb slots and asserts all bounds (project_lumopt2_campaign_state.md).
118. **An out-of-bounds seed kills the job in ~60 s after queueing behind everything** → `parametrization.py:674 _check_params`; the shape is always the same — a spec that FREEZES something (`free_comb=False` ⇒ comb bounds collapse to ±0.001 nm) combined with a seed or DETUNE point that moves it (`BEST_T9636` carries comb r = 80.1386; `run_adjoint_only`'s detune=1 sets the centre post to 100.0) → reproduce the runner's exact vector locally (seed → detune → clamp) and check against `param_bounds(spec)` (CLAUDE.md §5, `gates/predispatch_check.py`).
119. **Scene-snapshot references have CRLF while the snapshot writes LF** → a byte diff reports a difference that is CRLF-only → compare with `tr -d '\r' | md5sum` (project_antineedle_comb_stageP.md).
120. **`debug_fsp_compare/scene_snapshot.py` crashes on the two-device config** → `UnicodeEncodeError` printing a Greek delta to a cp1252 console (pre-existing, at `bragg_device.py:693`) → run as `PYTHONIOENCODING=utf-8 python …` / `PYTHONUTF8=1` (project_getent_false_negative.md).
121. **A generated runner written in cp1252 broke on the cluster** → the generator now writes UTF-8 (project_lumopt2_campaign_state.md).
122. **`setresource("FDTD", 1, "processes", N>1)` returns no error and reads back `'1'`** → silently rejected at the license tier; `addresource` grows rows but only gives parallel CAPACITY (1 vs 2 resources: 161.9 s vs 161.6 s for the same sim) → multi-GPU per sim is blocked project-wide, on every cluster (project_athena_multigpu_blocked.md).
123. **Conversely, explicitly setting `setresource("FDTD", 1, "processes", 1)` BREAKS the sim** → FDTD aborts after ~2 s without computing the modal port expansion; the default is already 1 (project_lumopt_adjoint_bug.md).

### Optimizer and gradient traps

124. **lumopt v1's adjoint gradient was 10–100× off FD** → fixed by a 4-fix stack that brought `vec_error` 11.40 → 0.144: (a) keep `w(λ)` explicit in the p=1 kernel (`_patch_porttransmission_weights`), (b) `frequency dependent profile = 1` on both ports (lumopt prints a stale "GPU doesn't support" warning but 2026R1 does), (c) `multi_freq_src=True` in the constructor, (d) an empirical **0.5×** kernel factor for the `t/(4P)` convention → plus `mesh_override_dxyz_nm = 25` on the freed region and `use_concurrent_adjoint_solves = False`; cavity_width residual stays 1.52 (project_lumopt_adjoint_bug.md).
125. **`lumopt` must be imported AFTER `measure_baseline()`** → importing it first poisons subsequent fresh `lumapi.FDTD()` sessions (FDTD runs in 1 s without producing port-expansion data) (project_lumopt_adjoint_bug.md).
126. **lumopt L-BFGS-B terminated after 1 iteration with FOM unchanged** → `scale_initial_gradient_to` defaulted to 0, so with FOM gradients ~1e-4 in scaled space the first step was ~0.034 nm physical = sub-Ångström, far below the 25 nm cell ⇒ eps unchanged ⇒ Wolfe line search rejects everything → set 0.25 (project_lumopt_scale_grad_fix.md).
127. **`opt.run()` returns params in SCALED [0,1] space, not nm** → the post-opt verification simulated `cavity_width = 299` instead of 798.6, which is why memory said "peak T went DOWN after optimization" → un-scale with `params_scaled / scaling_factor + scaling_offset` (project_lumopt_scale_grad_fix.md).
128. **★lumopt2's port adjoint carries a spurious 90° phase** → `port_fom.py:716` computes `e_adj * 1j*ω/4*conj(am)/P_src`; measured `dT_true(λ)` is ANTISYMMETRIC across the resonance while lumopt2's per-λ claim is a same-sign SYMMETRIC lobe (the quadrature); dropping the `1j` gives correlation **+0.991** with dT_true on all classes (shift p50, corr p24, comb-x p103) vs ~0.02 with it → fixed by an engine monkeypatch multiplying the scaled adjoint fields by (−1j·g), campaign C = **1.0561 + 0.1239i** (universal phase 6.71°/6.67°) (project_lumopt2_campaign_state.md).
129. **`validate_gradient` returns `(fd, adjoint, err%)` — FD FIRST** → a tuple-order misread inverted every recorded α for a whole session ("adjoint ×5-16 too SMALL" was actually ×5-16 too LARGE); pinned from source `utils/fd_grad.py:262` and cross-checked against the "Adjoint gradient at indices" log line (project_lumopt2_campaign_state.md).
130. **A retracted "breakthrough"** → "task 7 α ≈ 1.000 on all 7 params" was a self-comparison artifact: task 7's printed ADJOINT was compared against another job's ADJOINT mislabeled as FD; the adjoint is IDENTICAL (≤4e-4 relative) across naive/bc-only/bc+coloc, so both fix knobs changed nothing (bc_patch ≤0.04 %, coloc ~1e-6) (project_lumopt2_campaign_state.md).
131. **Per-class gradient calibration is scientifically DEAD** → α is OPERATING-POINT dependent, not a class constant: comb-x is α ≈ 1.3 at detune-1 but flips to α ≈ −10 at detune-2 (FD +8.4e-8 vs adj −9.0e-7); wrong-phase magnitude ∝ dλ_res/dp per parameter, which is why comb 1.3 / corr 5–8 / shift 16 / cavity 29 (project_lumopt2_campaign_state.md).
132. **`validate_gradient` compares the penalty-WRAPPED adjoint against RAW FD** → the gate's shift Re "−0.0352" IS the elongation penalty gradient, not physics (true naive Re ≈ 0 for all 3 classes = pure quadrature) → add `PEN_GRAD` back, or gate raw-vs-raw (project_v2_width_gradient_plan.md).
133. **The asymmetry runs the other way too** → FD does NOT include the κ-penalty while the adjoint print DOES (`attach_penalty` asymmetry) → any α reading at an out-of-deadband point must purify only the adjoint side; and detune points must stay INSIDE the deadband (at ρ = 1.092 every corr entry carried −3.204e-4 of penalty on BOTH adj and FD) (project_lumopt2_campaign_state.md).
134. **lumopt2's auto dp for FD-through-mesher is far below the mesh** → auto dp ≈ range·4.9e-4 ≈ 0.1 nm on our bounds vs dx = 50 nm → use explicit dp ≈ 1 nm (comb 2–5 nm) (project_lumopt2_igum.md).
135. **"conformal variant 0" staircases the grid-aligned tooth dEps** → tooth adjoint/FD scale 0.07–0.26 (4–14× too small) while comb cylinders were fine (0.77–0.98) → lumopt2 overrides to PVA so the adjoint works, at the cost of a ~5 % offset from the conformal spec convention (project_lumopt2_campaign_state.md).
136. **★A math gate is not a plumbing gate** → job 137267 died at 2:03 h on `IndexError: invalid index to scalar variable` at `self.fct = lambda x: anp.abs(x[0])[i_lo]` while the math gate passed at 0.0034 %: the fct's `x` is the FLAT vector `[T(λ_0)…T(λ_{n_wl−1}), softW]`, not a list of FOM entry results (which is why `x[-1]` is the width) → correct selector `lambda x: anp.abs(x[i])`; new <1 s gate `gates/gate_lam_chain_plumbing.py` asserts a one-hot jacobian AND that the old broken form still raises (project_v2_width_gradient_plan.md).
137. **A second defect in the same fix would have doubled peak RAM** → stashing `gfields_Tlo/Thi` alongside `gT` + `gfields_W` holds FOUR field sets, and the double pass alone already OOM-killed a 160G job at 501 λ (job 137012, exit 137) → convert each selector pass to its 191-float parameter vector immediately and `del f` before building the next (project_v2_width_gradient_plan.md).
138. **The FD gate itself OOM'd (exit 137)** → a multi-entry FOM holds a FULL region field array per entry (`base_fom.py:473-511`), and `get_fields_at_wavelengths` fetches the FULL λ grid then slices (`fdtd_session.py:1355`), so port entries pay full-grid cost even at zero jacobian → 151 λ points + 250G (project_v2_width_gradient_plan.md).
139. **★`profile_line` never integrated over y, so EVERY logged σ and FWHM is VOID** → it flattened (y, λ) into one axis and indexed with the LAMBDA index (always < n_lambda), so it ALWAYS returned y-row 0; `field_profile` is a Z-normal plane of y span 1.5·width_wide (~1.5 µm), so row 0 sits ~0.75 µm off axis in the evanescent skirt → T / λ / Q_L / Q_i / R / loss are PORT quantities and stand; every width, every σ wall calibrated from them, and every "in-band" label is void (project_lumopt2_campaign_state.md).
140. **★The width gradient was taken at a STALE, FROZEN wavelength (defect #19)** → `make_func` pins the single-λ twin `field_profile_adj::wavelength center` to `spec._wg_lam_track` "as A CONSTANT to autograd (zero Jacobian row, zero dEps)" (`lumopt2_design.py:570-578`), so `d(λ_pk)/dp` is structurally absent; AND `_wg_lam_track` is set in the log callback AFTER the eval, so eval N uses eval N−1's resonance (eval 0 falls back to `scan_center_nm`) → measured divergence `softw_um` vs `softw_adj_um` grew 6× in ONE iterate; unfixed cost ≈ +0.15 µm over 30 iterates, larger than the entire 0.10 µm margin (project_v2_width_gradient_plan.md).
141. **The naive λ-gradient stencil was 49.4 % low** → central difference of ∂T/∂p over a SECOND difference of T at k = round(0.5·fwhm/dl) = 20; the closed-form error is exactly `1/(1+x²)` with x = h/g (verified to the digit) → the MATCHED pair `gLam = −(g_hi − g_lo)/(T'(λ_hi) − T'(λ_lo))` is exact for any h and any symmetric lineshape (amplitude drift included), so a WIDE stencil is better (0.0034 % at k = 20 vs 49.38 % naive); the "factor 2" was coincidental, not structural (project_v2_width_gradient_plan.md).
142. **A guard clamp still produced a negative index and numpy WRAPPED to the far end of the spectrum** → with i_pk ≤ 1 or ≥ len(wl)−2 the k-clamp gave i_lo = −1 and the stash read `T[i_lo−1]`, silently building gλ from the wrong band edge → guard `1 < i_pk < len(wl) − 2`, swept all 501 peak positions to 0 out-of-range reads (project_v2_width_gradient_plan.md).
143. **A residual δ-leak of 0.60 % is structural** → the argmax INDEX sits ≤ dl/2 off the true λ₀, leaking ∂A/∂p at O(δ), h-independent → removable only by FITTING λ₀; **required numerics: ≥~40 spectrum points per spectral FWHM** (campaign 10 nm/501 pts = 20 pm vs FWHM 810 pm = exactly 40) → never widen `scan_width_nm` without raising `n_wl_points` (project_v2_width_gradient_plan.md).
144. **The IFT denominator can blow up** → `dλ_pk/dp = −(∂²T/∂λ∂p)/(∂²T/∂λ²)`; near a flat/merging/critically-coupled peak the curvature → 0 ⇒ unbounded gain; and for a Fano lineshape the T maximum sits at detuning δ = 1/q, NOT at the QNM frequency, so `∂T/∂λ = 0` can have a second root a stencil can hop to (reference_method_lit_check_2026-08-27.md).
145. **★The width wall was RANK-DEFICIENT** → corrugation was priced by mean(corr) only (identical gradient on all 25 teeth), shifts by total elongation only, cavity `wcav` not priced at all: ~48 of ~50 directions unpriced; the see-saw direction sat in the null space (the wall predicted −0.82 µm for a move whose MEASURED width change is −0.015 µm = 2.2× half-band error on the winning move) → uniform-seeded campaigns pinned at the band edge near T 0.917; fixed with 3-block per-tooth slopes FW_TOOTH_W (inner-8 −2.925e-3, middle-9 −2.384e-3, outer-8 −0.268e-3 µm/nm·tooth; 8(−2.925e-3)+9(−2.384e-3)+8(−0.268e-3) = −0.047000 = FW_A_MCORR exactly) (project_v2_width_gradient_plan.md).
146. **The width constraint explicitly PERMITTED a fifth of width growth** → `RHO_DN = 0.95` let ρ fall 5 %, and 5 % × (74.3/17.1) = **+21.7 % FWHM**; the design used 2.8 % of that 5 %, and σ's +2 % band then rubber-stamped it because σ cannot see ρ → nobody had ever converted the ρ band into microns (project_lumopt2_campaign_state.md).
147. **★The optimizer reconstructed a parameter the user had explicitly excluded** → Σshift elongates the cavity by 2Σs (+255 nm ≈ half a period by the walk's construction) → λ +2.6 nm → deeper mirror penetration → σ +9.6 % at fully compliant ρ 0.9888; i.e. the shifts' SUM is a linear combination that rebuilds the cavity-LENGTH knob → fixed with BETA_ELONG = 1e-5/nm², deadband |2Σshift| ≤ 120 nm; the analytic ρ proxy missed it because it models width as 1/κ with κ ∝ corr, blind to detuning-driven penetration (project_lumopt2_campaign_state.md).
148. **The restart then resumed AT the violator** → `_best_from_log` had NO width filter, and the violator was the FOM-max row, so each WidthTrip restart re-selected it — a ~1.5 h/cycle burn loop → `_best_from_log(…, sigma0_um)` now selects the best σ-COMPLIANT row and takes λ from that row (project_lumopt2_campaign_state.md).
149. **★A corrected run would have silently resumed the CONTROL** → `run_campaign` cold-start-resumes via `_best_from_log` reading `<out_dir>/<label>_evals.jsonl`, and task 41's label `lumopt2_v2_proj_toy` was the SAME label the control had written under, so the corrected run would have started at the control's iterate-2 point instead of the uniform seed, destroying the comparison and burning ~9 GPU-h (two jobs carried the flaw) → bump the label per attempt; the label IS the resume key (this trap struck 3×) (project_v2_width_gradient_plan.md).
150. **Pushing a constant to disk while a job runs creates a loaded-vs-disk divergence** → `RHO_UP` 1.01 was pushed AFTER launch, so running drivers enforced the LOADED 1.02 (proof: an accepted σ ratio of 1.0121 with no WidthTrip); the retroactive hazard is that on any job-level reload `_best_from_log` filters the OLD log with the NEW 1.01 and discards every >1.01 best → do not re-tighten a band mid-campaign (project_lumopt2_campaign_state.md).
151. **`trust_nm` protection silently vanished on resume** → `param_bounds` centred the clamp on `seed_params(spec)` (the module seed) while resume starts from the log's best row; for the bare lane the module seed has shifts AT 0 (an edge) → edge-fallback → FULL BOX → `run_campaign` now re-seeds `spec.seed_override = best["params"]` per attempt when `trust_nm` is set (project_lumopt2_campaign_state.md).
152. **`trust_nm` is a no-op when the seed sits on a physical bound** → `param_bounds` deliberately SKIPS the clamp there, so `trust_nm={"shift": 2.0}` left the uniform seed's shift bounds at [0, 200] unchanged (verified by printing the bounds) → and tightening `shift_bounds` is forbidden by standing rule (project_v2_width_gradient_plan.md).
153. **L-BFGS-B's first step is UNIT-NORM IN BOUNDS-SCALED SPACE** → free blocks with wide bounds get slammed by a fraction of their full range on the first probe regardless of seed quality: 2Σs 130.6 → **504.2 nm**, σ 17.749 → 19.888 µm (ratio 1.1369), FOM −7.92; the bare lane did the same (σ 21.3) → each rejectable probe costs ~1.7 GPU-h with maxls = 4 (project_v2_width_gradient_plan.md).
154. **Making the wall physically accurate KILLED the line search** → `fw_curve` is correct but much steeper than the linear wall (e = 287 → −793 vs −51), and with wide shift bounds scipy aborts with `ABNORMAL_TERMINATION_IN_LNSRCH` after ~6 of 100 evals (job 136640, 8 h 16 m) → fix = a SATURATING hinge `band = cap·band/(cap+band)` (rational, NOT tanh — tanh's cosh² derivative overflowed at e ≥ 163 with a measured RuntimeWarning); `fw_pen_cap = 2.0` since FOM is O(0.7) (project_v2_width_gradient_plan.md).
155. **A "zero-step" scare was scipy re-evaluating x0** → "FOM eval #2" returned a BIT-IDENTICAL FOM at bit-identical params 52 min after the baseline, costing one forward+adjoint (~1.7 h) → the discriminator is the next "Iteration 1: FOM =" line; centring trust bounds makes the scaler round-trip bit-identical and also removes this duplicate-x0 tax (project_v2_width_gradient_plan.md).
156. **A deadband gives ZERO gradient inside, so the quasi-Newton model cannot learn the wall** → every iteration's first trial step overshoots into it (2Σs probes 338.8 → 220.8 → 744.9) and is rejected: probe:accept ≈ 1:1, ~1 wasted solve per iteration (project_lumopt2_campaign_state.md).
157. **A noise-slack filter licensed a downhill drift** → `fom > acc.fom − slack` with `acc` overwritten on every accept means the REFERENCE WALKS DOWN with it; at slack 1.5e-3 the last 4 accepted iterates drifted 0.71832→0.71647 → anchor the slack to `fom_best` seen this run (`fom_ref = max(acc.fom, fom_best)`), updated AFTER the test (project_v2_width_gradient_plan.md).
158. **★Reuse staleness is ANGULAR — it scales with TRAVEL, not iterate count** → `k=5` was justified from a 0.685°/10 nm probe, but that is per 10 nm of TRAVEL; with the cap growing 25→38→57→60 nm while reusing 3 deep the stored gradient was ~180 nm stale ≈ 12°, far past the ~2.8° assumed → added `wgp_reuse_travel_nm = 40`; general lesson: when a knob is validated at one operating scale, re-derive it in the units the physics uses before combining it with a knob that changes that scale (project_v2_width_gradient_plan.md).
159. **A penalty-era failure response fought the optimizer for hours** → under the projection, width is steered by the STEP, so a width trip is a step-size failure; ratcheting `corr_max` (451→429→407 nm) never fixes the overshoot (d1u proved it 4× with zero progress) → under `wgp_ns2` a trip halves the persisted cap and forces a fresh width row, leaving corr_max alone (project_v2_width_gradient_plan.md).
160. **`dw_pred` is NOT a prediction to test against** → `dw_pred = gW·dp` is ~0 BY CONSTRUCTION in a `climb` step, so comparing measured dW to it divides by an intended zero (the resulting r = −52 is an artifact, not a C_field phase error) (project_v2_width_gradient_plan.md).
161. **Autogain's guard fired in the wrong phase** → `if abs(dW_pred) > 1e-4` let the CLIMB phase through, where dW_pred is a designed zero, so wgain would be slammed on noise and the negative-ratio "PHASE error" trip would cry wolf every iterate → gate autogain on `phase == "restore"` (the only phase whose step is built along ∇W), not on |dW_pred|; also an earlier claim that climb is orthogonal to ∇W "by construction" was WRONG (climb is `alpha·D·gT`, UNPROJECTED at :2154; RIDE is the orthogonal one at :2162) (project_v2_width_gradient_plan.md).
162. **A `wg_dwdlam = 0.3655` constant had no reproducible provenance and a naive re-derivation gave 0.5904 (61 % off)** → the filter rule is load-bearing: unique (λ, W) pairs AND `fom > 0.5·max(fom)`; lumopt2 re-logs the accepted point per restart segment (W 18.5076 appears 3×) and ONE out-of-band probe (fom 0.194 at W 19.53) pulls the slope 0.366 → 0.59 alone; do not pool baselines (different intercepts ⇒ 0.288); and the slope is NOT universal (0.300 seesaw vs 0.365 uniform, ~20 % spread) (project_v2_width_gradient_plan.md).
163. **A fitted width surrogate's coefficient was collinearity-contaminated** → the 12-row fit gave `SIG_A_WCAV = 0.109 µm/nm`, but a clean experiment (wcav +13.4 nm at FROZEN shifts) showed σ FLAT ⇒ the true coefficient is ~0 → the live job ran the loaded 0.109 and merely over-taxed wcav (conservative); and the earlier attribution of the λ 1559→1565 drift to cav_w was WRONG (wcav moved λ by 20 pm) (project_lumopt2_campaign_state.md).
164. **The σ̂ surrogate under-predicts (i.e. under-penalizes) outside its fit range** → predicted 19.264 vs measured 19.888 µm at the excursion, 4.7× outside its range → do not trust it far from the anchor (project_v2_width_gradient_plan.md).
165. **A width wall anchored on ONE seed falsely rejects compliant designs in another basin** → seedB2 ev5 was T 0.9591 at MEASURED ratio 1.0198 (IN BAND, better than its best) but σ̂ said 18.158 ⇒ penalty 0.528 ⇒ rejected; coefficients fitted on seed A do not transfer, and the anchor only re-zeroes on restart → re-anchor on EVERY ACCEPTED iteration (project_lumopt2_campaign_state.md).
166. **Two independent proxy walls on two parameter blocks forbid exactly the σ-neutral trades** → the constraint is ONE scalar (σ) but was enforced by an elongation deadband on shifts plus a ρ deadband on corr, structurally prohibiting the compensating between-block move; measured: at fixed σ, T gains ~+1.2e-4 per nm of 2Σs (payback Δρ = 9.6e-4/nm costs half the shift gain) ⇒ ~+0.01 T per +100 nm, with no bound nearby — and the tangent probe demonstrated +0.0127 T in ONE probe vs +0.0003 in the walled campaign's last 5 evals (project_inverse_design_cost_function.md, project_lumopt2_campaign_state.md).
167. **Penalties also create a FEASIBLE-PATH TRAP** → every iterate stays inside the allowed region, so designs reachable only via a temporarily-infeasible excursion are unreachable IN PRINCIPLE (project_inverse_design_cost_function.md).
168. **FOM values are not comparable across a wall change or across a penalty stack** → a new label is mandatory after changing the wall; and on the shift ladder, mcorr 357.95 ⇒ ρ 1.1014 charges every rung ~0.119 plus ~0.062 more at x1.5 (elong 198.9 > 120) → read T + `fwhm_env` only (project_v2_width_gradient_plan.md).
169. **`lams.ptp()` killed an 11.5 h campaign** → the ndarray METHOD was removed in NumPy 2.0 and the Athena container is numpy 2.x (IGUM is 1.x, which is why the same code ran for days there); the dwdlam refit engages at n ≥ 5 accepted points and every gate/smoke/toy ran ≤ 4 → `np.ptp(lams)`; general rule: a count-triggered branch needs a gate case AT K, and for every new conditional feature list its trigger conditions and check on paper that the validation run reaches them (project_v2_width_gradient_plan.md, CLAUDE.md §5).
170. **A reuse smoke was structurally unable to test its own feature** → the eligibility gate (|W − tgt| ≤ marg) could NEVER open at the surrogate's W (1.9 µm off target) → a smoke must assert the feature's own log marker fired, and the dispatch note must state which iterate is expected to trigger it (CLAUDE.md §5).
171. **An exception-based recovery path never engaged because lumopt2 double-wraps** → `scipy_optimizer.py:583` raises without `from e` and `optimization.py:852` wraps fct exceptions in RuntimeError, so `except RecenterNeeded` never matched and the job exited instead of recentring (seedB 54309 died this way) → catch RuntimeError too and unwrap BOTH `__cause__` and `__context__`; replicate the third-party raise chain locally before trusting any except-and-recover design (project_lumopt2_campaign_state.md).
172. **★My own h5 cleaner deleted the FORWARD field mid-gradient** → `Can not find result 'E' in field_profile_adj` on two resume incarnations; pass 1 (`-mmin +30`, keep newest 2 `*_output.h5`) deleted the fwd h5 because a slow resumed iterate (40 min adjoint + 49 min assembly) leaves fwd >30 min old while port-adj and width-adj h5s rank newer → `-mmin +240`, keep newest 4; a cleaner's age floor must exceed the longest live need-window and its keep-count must exceed the live-file count (fwd + 2 adj) (project_v2_width_gradient_plan.md).
173. **The same cleaner could not free space at all** → it kept the newest TWO `*_output.h5` **per study dir**, but every `*_files` dir holds only 2–3 files, so "keep 2" keeps ~everything: 22 study dirs × ~4 GB = 101.2 GB, quota at 289G/300G; its log was 0 bytes because it prints nothing when it deletes nothing → pass 2 drops scratch from any `*_files` dir whose newest h5 is >24 h old; **do NOT diagnose with `pgrep`** — it is a CRON job, so `pgrep -af "[h]5_roll_clean"` is empty even when working (misread as "janitor died" twice) (project_v2_width_gradient_plan.md).
174. **An "adjoint that ran in 52 min" had ALL-ZERO fields** → h5 forensic: the width-adjoint file's fields are exactly zero and the fwd twin plane has Ex = Ey = 0.0 (parity) — an import source at z = 0 CANNOT inject the TM adjoint, so the whole "GPU width-adjoint proven" claim was a source-less solve → guard `check_import_src_injects` raises on a dead source; keep-forever FD reference `[−0.00365, +0.01825, +0.02026]` (project_v2_width_gradient_plan.md).
175. **The GPU FieldRegion adjoint crashes on a ZERO dimension, not on size** → the shipped CUDA plugin validates both "Grid Dimensions … exceed the device limit" AND "include one or more ZERO values. All dimensions must be nonzero", and our twin is "2D Z-normal" with no z span (engine:1075); the GPU-unsupported list has NO field-region entry ⇒ unguarded crash path, and monitors are fine on GPU (DFT kernels handle singleton dims) — only the source path crashes (project_v2_width_gradient_plan.md).
176. **The CUDA error surfaces ~22 MINUTES LATE** → the engine meshes on CPU first, so "dies in seconds" is wrong → judge a rung on EXIT CODE, never on elapsed time (a premature "it launched" claim had to be retracted) (project_v2_width_gradient_plan.md).
177. **Chasing that bound on FULL-DEVICE rungs cost ~6 GPU-h and an evening** → four jobs at 45–70 min each (136799/136826/136869/136907) bisecting a CUDA kernel-launch bound that has nothing to do with the grating → `gpu_probe.py` answers the same question over 12 sizes in one short job; the threshold was still un-bisected between 3,696 (pass) and 29,568 (fail) cells, which matters a lot (4k budget ⇒ ~16 tiles ≈ 6 h, killing the speed win) (CLAUDE.md §5, project_v2_width_gradient_plan.md).
178. **Fitting `C_field` on a CROPPED region is invalid** → the region's softW is a different functional than the FD's, so the measured ~0.400 ratio may be C or may be the crop → fit at the production region; and a fitted COMPLEX constant is a documented bug signature (arg ≈ −16.4° is an intermediate phase, which comes from time-origin/reference-plane mismatch, not from normalization) → test invariance across ≥2 λ, ≥2 mesh sizes, ≥2 monitor spans, ≥2 device lengths (project_v2_width_gradient_plan.md, reference_method_lit_check_2026-08-27.md).
179. **Cutting λ points is not a speed lever** → 501 → 151 saved only **11 %** (3136.7 → 2795.9 s) → do not trade numerics for it (project_v2_width_gradient_plan.md).
180. **The parameter count is not what it looks like** → `N_PARAMS = 191` always, but `param_bounds` gives PINNED blocks sliver bounds (±1e-3 nm), so the FREE dimension is 76 (shifts free) or 51 (shifts frozen); dEps still differentiates all 191 slots, spending ~60 % of its 397 s on columns that cannot move (~5 % of iteration time), and removing the slots caused a bounds-rejection job kill (project_v2_width_gradient_plan.md).
181. **A soft-level-set width is only piecewise-smooth** → its gradient concentrates on the half-max CROSSINGS, so a shoulder or second lobe makes the crossing jump discontinuously and `∇softW` rotate non-smoothly, abruptly rotating the feasible manifold → the `fwhm_env` vs `softW` anchor residual should be an ABORT condition, not a watched number (reference_method_lit_check_2026-08-27.md).
182. **lumopt2 has NO checkpoint/resume and does not trap SIGTERM** → `scancel` loses in-flight state; recovery is copying the final-params block from `optimization.log`, or (ours) cold-start resume from `<label>_evals.jsonl` (project_lumopt2_igum.md).
183. **lumopt2's `SlurmRunner` is broken as shipped in R1.2 AND R1.3** → it imports `lumopt2.utils.lumslurm`, which does not exist (the real module is `api/python/lumslurm.py` one level up) → `import lumslurm; sys.modules["lumopt2.utils.lumslurm"] = lumslurm`; residual bugs: it marks jobs done WITHOUT verifying success, and `run_dependencies` launches only the first dependency (early return) (project_slurm_container_fixes.md, project_lumopt2_igum.md).
184. **lumopt2 API landmines** → `visualize_geometry`/`visualize_fom` open CAD windows and block on `input()` (never call on a cluster); `validate_gradient` and `fd_sweep_perturbation` call `plt.show()` (need `MPLBACKEND=Agg`); `JSONLogger` is not exported at package level; `FieldResults` is single-λ only with metric = Σ|E|² over the monitor (a spatial second moment is NOT expressible); broadband NON-port sources are rejected; a FOM monitor straddling a symmetry plane off-centre gives only a `logger.warning`; there are no fabrication constraints anywhere; `__version__` is `'0.0.0'` so R1.2 and R1.3 lumopt2 are indistinguishable at runtime; `ClosedCurve`'s docstring example (`from_path`, tuples) is STALE and will not run; `Topology` is undocumented and immature (project_lumopt2_igum.md).
185. **The bundled `lumopt` v1 is an Ansys fork with a BREAKING API** → `get_fom` → `(fom, fom_wavelength)` and `fom_gradient_wavelength_integral` → `(grad, grad_vs_wl)`, so third-party FOM classes written for classic lumopt break; its `porttransmission` hard-codes port names 'fom'/'source', lacks `adjoint_source_name` (AttributeError in the one_forward path) and rebuilds the λ axis assuming uniform spacing (project_lumopt_v1_ansys_fork.md).
186. **Lumerical-native PSO via `addsweep("Optimization")` silently no-ops in headless lumapi** → a readback after configuring showed `'type' = 'Values'` (not 'Optimization') with every optimizer property `None`; seven successive fix attempts each exposed a different silent rejection (`addsweep(1)` → "cannot find item 'opt_peak_T'"; a full LSF script via `fdtd.eval()` → `LumApiError: 'Failed to evaluate code'` before any execution); `getsweep(...,'parameters')` also fails the same way → use the Python-driven PSO instead (project_lumerical_native_pso_blocked.md).
187. **The PSO's parametric `.fsp` builder produced a DEAD TM device** → modal |S21|² ≈ 0.000829 flat for EVERY particle including one geometrically identical to the baseline, which the NORMAL builder reads as T = 0.945; all 132 particles tied at noise ⇒ PSO "converged" at gen 2 and the post-opt result (0.911) was below baseline (0.945) → `rebuild_per_particle=True` scores each particle via full `run_single_sim` (validated: 0.000829 → 0.9454); the parametric/FOM-.lsf/static-skeleton layer is adjoint-only scaffolding (project_tm_transmission_pso.md).
188. **A headline Δ was an unfair coarse-vs-accurate comparison** → the driver compared a COARSE baseline against an ACCURATE optimum (accurate reads ~0.018 below coarse), reporting Δ = −0.002 for a real gain (project_tm_transmission_pso.md).
189. **A sign error in a payback normalization penalized narrow rungs** (`retrim_decompose.py:69`) → ±0.004, no ordering changed, but it was found only by re-reading (project_v2_width_gradient_plan.md).
190. **A "converged" flat campaign was going BACKWARDS** → stage-4 FOM 0.7157612 (eval 3) → 0.7157579 (eval 9), i.e. −3e-6 over 13 h → the 10-iterate flatness rule was replaced by predictive stopping (`dT_pred = ∇T·step < 0.002` on 3 consecutive accepted iterates, or the trust cap pinned at its 2 nm floor) (project_v2_width_gradient_plan.md).
191. **A `plt`-free "MX-GRAD" style config diff appeared to prove the lumopt2 scene builds a different device** → the diff compared against SweepSpec DEFAULTS (showing polarization TE and pitch 500 nm, impossible for a TM corr-325 study), not against the real device → invalid, withdrawn; the genuine cause of the 5.3 nm offset was the mesher (project_lumopt2_campaign_state.md).

### Cluster, SLURM and license traps

192. **★`lmstat` returns −96 "lmgrd is not running" while the license is FULLY WORKING** → lmstat enumerates by the server's advertised FQDN `lumerical-lm.ece.technion.ac.il`, which does not resolve from Athena nodes, while real jobs check out BY IP via `ANSYSLMD_LICENSE_FILE=1055@132.68.48.51` / `ANSYSLI_SERVERS=2325@132.68.48.51` → open TCP ports 1055 + 2325 by IP + one real run are the reliable signals; job 115369 ran real 7-min solves right after preflight declared "down" (project_athena_lmstat_false_negative.md).
193. **A GENUINE outage looks different** → ports 1055/2325 CONNECTION-REFUSED from both clusters, and IGUM's lmstat gives `-15,570 Cannot connect to license server system` → the discriminator is refused vs open ports (project_athena_outage_2026-07-25.md).
194. **Reachability ≠ availability** → open ports only prove the server answers; the seat count is what kills runs (the pool oscillated 39–46/50 within hours) → probe the count FROM IGUM: `$LUM/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve` (CLAUDE.md §6).
195. **★License starvation has two completely different signatures** → IGUM (native): instant loud death, bare `in run:` + "Unable to checkout the requested HPC license"; Athena (container): **SILENT no-op** — `fdtd.run()` returns in ~1 s with no fields and the pipeline later crashes "Can not find result 'expansion for port monitor'" → check the log's `Simulation time` first (~1 s = license) (project_license_failure_modes.md).
196. **That port-expansion error has TWO causes** → shared-.h5 clobber OR a license no-op → same discriminator (project_license_failure_modes.md).
197. **★One GPU solve consumes SEVEN `lum_fdtd_solve` seats** → 6 solves = 42/50, so the ceiling is ~7 concurrent solves across ALL arrays and clusters; the 8th dies instantly with FlexNet −4 surfacing as a bare `LumApiError 'in run:'` (check the layout `*_p0.log`); launching 8 on top of 4 running license-killed tasks 2,3,5,7,12,13 (project_trench_n150_hscan_igum.md).
198. **The 6-concurrent ceiling is an UPPER BOUND, not a guarantee** → the pool is faculty-shared with invisible external consumers; 4 IGUM + 2 Athena tasks died on seats that "should" have existed, and our own queues being empty proves nothing (project_license_failure_modes.md).
199. **Raising the throttle at 37/50 seats cost more than the fan-out saved** → the rule is hold at ≥35 in use; 4 tasks died with the bare `in run:` signature (project_itai_hh_apodization.md).
200. **★IGUM tasks die at 60 s with an `ansyscl` error that is NOT seat starvation** → `Could not open 'fdtd': appOpen error: … did not produce the startup UUID within 60 s` plus `ANSYSLI exited or could not read server port ansyscl.<node>.<node>_<user>_261. No such file`, exit 1 — happens with 31/50 seats free and healthy lmstat, and the loud "Unable to checkout" text is ABSENT; the ansyscl client daemon is per-user-per-node and simultaneous starts race it → dispatch with `--max-concurrent=1` for anything opening a lumapi session; measured 1/4 dead at `%4`, then **4/4 dead within 17 s** at `%2` onto a node already hosting 2 of our jobs (a cascade), while tasks started ALONE minutes apart were fine (project_igum_ansyscl_startup_race.md).
201. **N tasks cold-starting an array in the same second also race the checkout daemon** → losers die instantly, survivors fine; recovery is a staggered `--array-tasks=<dead indices>` resubmit once the queue drains (runs are idempotent) (project_license_failure_modes.md).
202. **QOS `24h_1g` caps 100 submitted / 4 RUNNING per user** → arrays >100 tasks must go in chunks, and plain `squeue` collapses a pending array to ONE line → count with `squeue -r`; `sacct` TRIPLE-COUNTS array steps (divide by 3 or trust squeue + result-file count) (project_tm_scatterer_scan.md, project_scatterer_greens_program.md).
203. **QOS caps memory at 275G per job** → `sacctmgr show qos` gives `cpu=32, gres/gpu=1, mem=275G` for both `24h_1g` and `4d_1g`; 300G is REJECTED with `sbatch: error: QOSMaxMemoryPerJob` / `Job violates accounting/QOS policy`, and the deploy reports only `ERROR: sbatch failed.` unless you read the full output → use 256G as the practical ceiling; other caps: MaxJobsPerUser 4 (24h_1g) / 8 (4d_1g) / 3 (2h_2g, 12h_4g, 24h_4g) / 1 (72h_8g) (project_athena_job_memory_footprint.md).
204. **`scontrol update` on a pending task BYPASSES the 275G cap and accepts 300G** — submission does not (project_cladding_reflector_dispatch.md).
205. **`2h_2g` caps `mem=240G` PER USER, which silently serializes a 2-up array** → 160G running + 160G queued > 240G → PENDING reason strings name the limit but not WHICH pool it belongs to; a "queue contention" rationale was cited wrongly and had to be retracted (project_lumopt2_campaign_state.md).
206. **★`ARRAY_TIME` as an env override is SILENTLY IGNORED** → `athena.conf` is sourced by the deploy AFTER the env is set and PLAIN-ASSIGNS `ARRAY_TIME` (measured: passed 24:00:00, job got 23:30); `SBATCH_MEM` DOES work (read as `${SBATCH_MEM:-}`) → change times via conf knobs and verify with `sacct --format=TimeLimit`; the same conf trap applies on IGUM (project_slurm_container_fixes.md, project_q3db_predictive_engine.md).
207. **`SBATCH_MEM` was wired into the `--option2` branch only** → array sweeps OOM'd at the default; fixed by adding MEM_OPT to the option-3 sbatch (~line 1166) (project_cladding_reflector_dispatch.md).
208. **EVERY Athena GPU partition is `PreemptMode=REQUEUE`, `a100-public` included** → `PreemptType=preempt/qos` and the contrib QOS (priority 10000, MaxWall 7d) preys on every lane we have (12h_4g, 24h_1g, 24h_4g, 4d_1g, 4h_0g, 72h_8g), so there is NO preempt-proof lane; B4 lost 8.9 h to a contrib job → protection is resume, never lane choice; GraceTime is 10 min, `JobRequeue=1`, `MaxBatchRequeue=5` (project_slurm_container_fixes.md).
209. **The FDTD engine has NO checkpoint/resume** (CLI help checked) → a single long solve is ATOMIC and unprotectable by logging; the criterion for placement is DURATION, not mesh mode (an optimization-mesh solve ran 71.5 h) → re-check for engine checkpointing on every version bump (project_slurm_container_fixes.md).
210. **★Deploying ANY `--option3` study while another array has pending tasks rewrites the shared `data/sweep_list.txt` and kills those tasks** → hole-scan tasks 13–97 aborted with "SWEEP_INDEX out of range (file has 6 lines)" after the list went 98→4→6; the apodization study lost tasks when 48→14; `sacct` still reported COMPLETED for early ones, so read task LOGS; recovery = wait for drain, redeploy, resubmit the dead range with `--array-tasks=<lo>-<hi>` → fixed structurally by PER-STUDY lists `data/sweep_list_<study>.txt` + SWEEP_LIST export (project_tm_scatterer_scan.md, project_apodization_sweep_tm_te.md, project_lumopt2_campaign_state.md).
211. **Worse than dying: an in-range index silently runs the WRONG study's row** → and REQUEUE makes even "running-only" queues vulnerable, which is why the old absolute rule existed (CLAUDE.md §6).
212. **`rsync --delete` swaps shared engine/builder code under an in-flight study** → a REQUEUEd task then silently re-runs at different numerics; there is no sync-free flag (`--upload-only` still does the full rsync) → deploy from a STAGING COPY whose baseline is the remote (`rsync -a user@host:REMOTE_BASE/project/ "$STG/"`, overlay your file, prove the delta with `--dry-run --itemize-changes`), and md5 the engine + each live runner local-vs-remote before any parallel deploy (project_staging_deploy_concurrent_chats.md, project_lumopt2_campaign_state.md).
213. **Two invented deploy flags were silently ignored and submitted real jobs** → `--no-submit` produced stray 10-task array 133070 (which would have re-run every gate) and `--no-dispatch` produced duplicate campaign driver 54440 an hour later, while the legitimate `--upload-only` existed all along → both deploy scripts now ABORT on unknown flags; read the parser before passing a new flag and check the queue after every deploy (project_lumopt2_campaign_state.md).
214. **The deploy never aborted on a FAILED sweep-list build** → an `echo` clobbered `$?` before the check (classic bash trap), so a failed build submitted a fallback 1-task list instead (job 133394 died in 8 s on a missing top-level `SPEC`) → `BUILD_RC` captured at assignment, in BOTH scripts (project_lumopt2_campaign_state.md).
215. **An interrupted deploy had ALREADY submitted** → duplicate array 133882 appeared alongside 133883 and same-label duplicates RACE on shared files (the one collision per-study lists cannot prevent) → after any interrupted dispatch, CHECK THE QUEUE for a ghost submission (project_lumopt2_campaign_state.md).
216. **Athena's login node KILLS all user processes at ssh logout** → a nohup'd sleep died in seconds and the tmux server died too → run container surgery as a SLURM CPU job; `--wrap` is FORBIDDEN by the site `cli_filter_lua` (script files only) (project_athena_container_rebuild_pipeline.md).
217. **Jobs hang 10–20+ min at `INFO: Setting --writable-tmpfs (required by nvidia-container-cli)` on every node type** → usually /home OVER QUOTA (soft 300G / hard 330G; `quota -s` shows `NNNg*`), because writes block and the container overlay setup stalls; caused by ~130 rebuild-PSO sims (.h5 58.8 GB, .mat 104 GB, 426 .fsp) pushing home to 319G → delete disposable `.h5` (NOT the .mat) and deploy with `KEEP_H5=0`. **But the container hang is not always quota**: rtx6k-shared nodes n317/n318 hung at the same step under quota, on a new driver (595.45.04 / CUDA 13.2) incompatible with the container → rule out both (project_athena_quota_hang.md).
218. **lumopt2 NEVER cleans solver scratch** → a validation label alone reached 18 GB (fwd_default `.h5` dir 3.5 GB + adj/FD files); per-iter names are fixed so campaigns stay ~15–20 GB steady-state, but stale validation `_files` dirs accumulate ~4 GB per completed study (project_lumopt2_campaign_state.md).
219. **A field-monitor run's host RAM is monitor-driven, not domain-driven** → monitors OFF at 6× domain = ~3.5 GB; far-field + 2D fields = 55.4 GB; full 2D/3D field profiles = >100 GB; and a MONITORS-OFF ports-only sweep still OOM'd because port monitors store E,H on the full cross-section × freq points (converged box 6.8×8.8 µm at **7001** pts died OUT_OF_MEMORY where 3001–4001 pts ran fine) → memory ≈ box × points: Ybox6.8/3001 = 72 GB, Ybox16/3001 ≥ 133 GB (OOM at 128G), Ybox16/1501 ≈ 85 GB (project_athena_job_memory_footprint.md, project_scatterer_greens_program.md).
220. **A 16 µm-box run OOM'd in POST-processing, not the solve** → the sim finished in 52 min and `get_s_and_t_matrix`'s port mode-expansion over the big transverse box spiked host RAM to 132.8 GB against a 128 GB request (project_cladding_reflector_dispatch.md).
221. **IGUM cannot run containers at all** → no apptainer, no singularity, no module system; `docker-ce` is installed and the user is in the `docker` group but `docker info` is DENIED → every version bump is an extracted RPM tree we own; IGUM also has **no `rpm2cpio`** (Ubuntu, only `cpio`) → extract on Athena and tar-stream over the LAN with an md5 manifest (project_igum_cluster.md).
222. **IGUM's `slurmdbd` is DOWN** → `sacct`/`sacctmgr` give "Connection refused localhost:6819" while sbatch/squeue/scontrol work → use squeue + file mtimes, and `scontrol show assoc_mgr` for limits (project_igum_cluster.md).
223. **IGUM QOS names are not Athena QOS names** → `LUMOPT2_QOS=4d_1g` is an ATHENA name and gives "ERROR: sbatch failed" on IGUM, which needs its QOS to MATCH the partition (`qos-preempt`/`part-preempt`) plus `--account`; `--gres=gpu:N` is the IGUM form, not Athena's `--gpus=1` (project_v2_width_gradient_plan.md, project_igum_cluster.md).
224. **IGUM `part-preempt` compute nodes are BARE** → three successive 2-second/2-minute deaths: `libgfortran.so.5` (scipy), then `libXi.so.6` and the GL family, then a runtime-**dlopen'ed** `libxcb-dri2` that is INVISIBLE to `ldd` → copy the whole X/GL client family (libxcb*, libX*, libGL*, libEGL*, libgbm, libdrm, libxshmfence, libglapi, libwayland, libxkbcommon = 173 files) from the login node into `scilibs` and put it on LD_LIBRARY_PATH; `libglut.so.3` is absent everywhere (apt-get download + dpkg -x + symlink) (project_trench_n150_hscan_igum.md).
225. **`fdtd-engine -v` failing on `libglut.so.3` is NOT a broken install** → it means `scilibs` was not on LD_LIBRARY_PATH; test with the job scripts' own env (project_igum_cluster.md).
226. **★Automated polling got our key refused on IGUM** → monitor v2 raised IGUM connections from ~6/h to ~24/h and auth failed ~80 min later; ~45 min of ZERO contact restored it with no key change → budget ≤~3–6 ssh/hour per cluster, ONE connection per poll (fold the lmstat probe into the same ssh), and on ANY auth refusal stop all automated contact ≥45 min rather than retrying (retries deepen a ban) (CLAUDE.md §6).
227. **But a refusal is NOT proof of a ban** → a deploy succeeded at 08:05 BETWEEN refusals, which makes a re-arming ban implausible; IGUM's own sshd/home-FS flaps (slurmdbd was also down, and seedB had been requeued, indicating a node/service restart) → a full TCP timeout does not discriminate between a packet-dropping fail2ban and a host outage, so probing can only make the ban case worse (project_lumopt2_campaign_state.md).
228. **Cluster JOBS are unaffected by login-node auth** → seedB and the chained bare campaign ran on compute nodes throughout and afterok still fired; an outage costs visibility, not science → never panic-redispatch (CLAUDE.md §6).
229. **A monitor reported both clusters down simultaneously** → passive TCP probing (no ssh, no budget spent) diagnosed it as LOCAL: Athena's hostname failed DNS entirely (gaierror) while IGUM and the license server (raw IPs) timed out on port 22; three independent Technion hosts do not drop together → and monitor v1 had silently mangled a SINGLE-cluster outage because it only guarded the both-down case (project_lumopt2_campaign_state.md).
230. **A watcher that exits on "no jobs" exits on an ssh failure too** → only exit on ssh-SUCCESS + no jobs (project_scatterer_greens_program.md).
231. **`sbatch --export` is itself comma-delimited, so a comma-containing value gets TRUNCATED** → `TM_WIDE_SEEDS_NM=227.7,250` arrived as `(227.7,)`, and one seed made the secant's `pts[-2]` raise IndexError AFTER a completed 23-min GPU eval → never pass comma-containing values through `--export` (project_tm_wide_mode_corr.md).
232. **Empty `${VAR:-}` exports crash runners with `float("")`** → switch `.get(key, default)` → `.get(key) or default` everywhere env knobs are read (project_tm_wide_mode_corr.md).
233. **`TM_PITCH_NM` is forwarded as 500 by default and would clobber a study's pitch** → the wide-mode runner needed its own `TM_WIDE_PITCH_NM` added to the deploy export list (project_tm_wide_mode_corr.md).
234. **Results nest by STUDY dir (the module basename), NOT by label** → a wrong-path guess cost a diagnostic detour; and a glob on `results/elong_*/` matches the STUDY dir and finds no jsonl, which produced a false "tasks failed" alarm when tasks 1+2 had exit 0 all along → correct shape is `results/<study>/results/<label>/<label>_evals.jsonl` (project_lumopt2_campaign_state.md, project_v2_width_gradient_plan.md).
235. **Restrictive dir perms on remote `project/` can make rsync silently SKIP root `*.py` files** → jobs then crash <30 s after a config change on stale server code → `rsync --inplace` is the known fix; verify the itemized output actually updated the files you edited (CLAUDE.md §6).
236. **A DGX "successful" sim completed in 3.452 seconds** → R470 / CUDA 11.4 drivers predate the container's CUDA runtime; the `nvml_tramp` shim makes GPU DETECTION report `'GPU'` but the CUDA kernels never launch (0 MiB allocated) and Lumerical does not surface it as an error; post-processing then crashes on the missing port expansion → the symptom is a short `Simulation time:` followed by the port-expansion LumApiError (project_dgx_fdtd_gpu_broken.md; cluster retired 2026-09-14).
237. **A user `License.ini` with `domain=1` silently breaks lumapi** → on Zeus, `~/.config/Lumerical/License.ini` is read BEFORE the system one; launching the GUI on the head node creates it and lumapi then fails with "appOpen error: Failed to start messaging, check licenses" even with reachable servers → export `ANSYSLMD_LICENSE_FILE`/`ANSYSLI_SERVERS` in the job scripts (project_zeus_lumerical_license.md).
238. **A dead license server is SILENT** → the CAD acquires and caches a `lumerical_main` seat so layout operations all succeed, but the time-stepping engine needs another seat and `fdtd.run()` simply returns with empty monitors and status 0; jobs finish in 17–30 s → the fastest decisive test is a tiny LOCAL run (a trivial 2D sim returning in 0.00 s with no results), not lmstat (project_technion_license_outage_2026-05-19.md).
239. **`athena.technion.ac.il` hung post-auth for >1 h** (ping + auth OK, shell/scp never respond) → `dgx-master.technion.ac.il` (132.68.1.201) shares the SAME /home and works as a file-access fallback door (do NOT dispatch jobs from it) (project_scatterer_greens_program.md).
240. **Hostnames do not resolve without the Technion VPN** → use the `~/.ssh/config` aliases (`ssh athena`, `ssh igum` pinned to 132.68.58.101); `igum.technion.ac.il` does not resolve and `igum.ece` fails the host-key check (project_lumerical_versions_and_athena_ansys_gate.md).
241. **A stale IGUM host key in Athena's `known_hosts`** after IGUM's key rotation broke the LAN tar-stream → verify the fingerprint out-of-band, then `ssh-keygen -R 132.68.58.101` on Athena (project_athena_container_rebuild_pipeline.md).
242. **Athena's native `/apps/ansys` looked EMPTY** → it was a silent permission-denied (root:ansys mode 750, `getfacl other::---`); after the group was granted it turned out to hold only Ansys 2025 R1 (v251), a full generation older than the container → going native would be a DOWNGRADE plus a numerics change (project_lumerical_versions_and_athena_ansys_gate.md).
243. **A loud UCX/InfiniBand `ucs_handle_error` backtrace at `MPI_Finalize` on athena-post is COSMETIC** → the version string printed first and the engine md5 gate passed (project_athena_container_rebuild_pipeline.md).
244. **The VPN speed is not a constant** → measured 0.19 MB/s once (making a WSL round trip ≈ 13 h) and **10.6 MB/s** on another day → measure before planning an overnight push; and the 5 GB `.sif` must never cross the VPN (project_athena_container_rebuild_pipeline.md).
245. **The engine-bump canary is expensive** → ~2 h 40 m of GPU per cluster (~5.4 GPU-h total) for ~175 µm of grating × 8.0 × 8.8 µm at dx 50 nm ≈ 50 M cells with Q ≈ 14k needing ~16 photon lifetimes → the log's "Max time remaining: 44 hrs" is the no-shutoff worst case and is NOT a warning sign (auto-shutoff fires near 5.5 % complete); ring-down τ ≈ 13 ps vs Q/ω = 11.5 ps is a free physics check from `*_p0.log` at ~2 % (project_athena_container_rebuild_pipeline.md).
246. **Order matters in the container pipeline** → LAN-stream the extracted tree to IGUM BEFORE launching the build job, because the build `mv`s the stage into the sandbox (project_athena_container_rebuild_pipeline.md).
247. **Binding the HOST `/etc/passwd` into the container BREAKS LDAP users** → apptainer injects the login user into the container's own passwd and the host file lacks LDAP entries → bind a MERGED file (container's own + `getent passwd slurm`); also file-binds into `/usr/lib64` do NOT work for dlopen'ed plugin deps — bind a DIR on LD_LIBRARY_PATH (project_slurm_container_fixes.md).
248. **A completed task can be REQUEUED after writing a complete result** → 130184_3 was requeued by SLURM after writing its .mat (harmless overwrite re-run) (project_antineedle_comb_stageP.md).
249. **A campaign that hits its walltime leaves no `_best.json`** → IGUM 55801 died on TIME LIMIT after 12 evals having never reached the completion path (one adjoint = 3107 s, so ~1.5–2 h per gradient iteration); resume protection worked (the eval log persisted) but the walltime was undersized → long campaigns need `4d_1g`-class QOS (project_lumopt2_campaign_state.md).
250. **A job in `TIMEOUT` state can still be a success** → B4 converged then hit walltime; the TIMEOUT state is cosmetic and the eval log holds the full trajectory (project_lumopt2_campaign_state.md).
251. **Multi-node / `24h_16g` buy us nothing** → a Lumerical sim is capped at 1 GPU by the license tier, so the lever is 1-GPU/task arrays and the binding caps are QOS `24h_1g` (4 running) plus license seats, not hardware; what DID change in 2026-09 is queue depth — `a100-public` is 5 nodes / 40 A100 (project_athena_multigpu_blocked.md).
252. **`a100-staging` is idle but unusable** → `AllowAccounts=admins-projects` (project_v2_width_gradient_plan.md).
253. **IGUM and Athena reproduce each other exactly, so cross-cluster re-anchoring is waste** → IGUM ctrl = Athena ctrl to 0.0001, h350 to 0.001, h4000 to 0.0015; and the R1.3 canary reproduced the stored anchor in every printed digit on Athena (T Δ 1e-6 on IGUM) → a cluster switch alone never justifies a control re-run; identity = engine version + §2 numerics + spec params, NOT cluster (project_trench_n150_hscan_igum.md, project_athena_container_rebuild_pipeline.md).
254. **Wall-clock from a shared-node array is NOT usable for cost scaling** → 3–4 tasks shared `ece-efrats5`; per-node speed spread is real (same task: RTX PRO 6000 2.2 h < A4500 3.5 h < 2080Ti 4.8 h) (project_tm_nladder_surrogate.md, project_trench_n150_hscan_igum.md).
255. **Cross-hardware scatter is ~0.002 in T, which is the dx=50 floor** → a re-measure of identical params on a different node read T 0.9375 → 0.9357; steps under that are not results (project_lumopt2_campaign_state.md).

### Data, file and path traps

256. **A `--results-no-fsp` download FAILED SILENTLY** (connection reset) and was only caught a day later → recovered via direct `scp`; all 7 .mat then verified local (project_antineedle_comb_stageP.md).
257. **`bash athena/deploy_athena.sh --results-no-fsp` HANGS forever (0 bytes) when run as a background task** → it blocks on an interactive prompt with no stdin → for a few files use `scp` directly; **remote brace expansion like `result_{a,b}.mat` does NOT expand through scp** — loop over names (project_getent_false_negative.md).
258. **A killed mid-stream `--results-no-fsp` tar left a STALE old-index local file** → it showed 1571 nm from the 1.9963 era → re-scp the specific file (project_tm_pitch_redo_1p97.md).
259. **`rsync` from Windows Git Bash needs `/c/Users/...`** → `c:/Users/...` is read as `host:path` (project_scatterer_greens_program.md).
260. **A remote result .mat was 1.2 GB with no variable >16k elements** (local slim re-save is 98 KB, siblings ~500 KB) — unexplained, still open (project_te_q3db_20um.md).
261. **A claimed metrics file was never written** → `~/n150_metrics.txt` did not exist (a prior session died mid-extract); the claim had to be corrected (project_scatterer_greens_program.md).
262. **A cleanup deleted the only copy of a 1.86 GB raw `.h5`** → a lifeboat HARDLINK saved it (`SAFE_*.h5`) (project_tm_h200_w1800_study.md).
263. **Full field `.mat` files are ~650 MB each and the link is ~0.5–1 MB/s** → 2.5 GB was pulled for 4 images before switching to server-side slicing (login-node `python3` has numpy/scipy); a figure needs one plane at one λ (~1 MB) (CLAUDE.md §6).
264. **A `.fig` saved with `Visible='off'` opens BLANK** → set Visible on before `savefig` (fixed in `plot_scatterer_greens.m`) (project_scatterer_greens_program.md).
265. **Convergence-study `.mat` results are keep-forever data** → a lost TE convergence set forced a full rerun; the TE convergence data in fact lives only in a hand-curated Excel (`mesh_convergence_results.xlsx`), not as a checkpoint (CLAUDE.md §7, project_tm_convergence_study.md).
266. **Stored logs can carry VOID columns that still look like numbers** → every pre-2026-08-18 σ/FWHM in the lumopt2 eval logs, the trade line `T = 0.89265 + 0.01549·dFWHM`, the per-lever efficiencies (0.0203 vs 0.0116 T/µm), the "seedB ev1 is +0.0156 above the line" ranking and the FWHM_hat ratios 1.17–1.19 are all built on them → keep as a record of reasoning; re-derive before citing; stored `.mat` files carry `field_x`, `field_energy_density_1D`, `field_envelope_1D` and `fwhm_m`, so any width question is answerable OFFLINE forever (project_lumopt2_campaign_state.md).
267. **A generic plot script hardcodes its own OUT_DIR and clobbers another study's PNGs** → `plot_side_by_side_maps.py` hardcodes `OUT_DIR=side_by_side_coupling` and plots COUPLING, not peak T (project_side_by_side_coupling.md).
268. **Splitting TE/TM by `endswith('_te.mat')` breaks** → a `_smp` tag follows, so split on the `_te_`/`_tm_` filename TOKEN (project_tm_period_match_te.md).
269. **The stored far-field `E2` array has rows = ux** → transpose for `pcolormesh`; the same class of bug was fixed in `farfield_export.py` (top_monitor was H=uy, should be H=ux) and `analyze_farfield.py` (missing `.T` on three contourf calls) (project_scatterer_greens_program.md, project_visualization_conventions.md).

### Tooling traps (Python, MATLAB, Windows/PowerShell, ssh, the deploy script)

270. **★`getent hosts` is a FALSE NEGATIVE under Git Bash on Windows** → it is a glibc/NSS tool and does not consult the Windows resolver; it reported even `google.com` unresolvable while the network was fine → a "wait for the network" watcher armed on it would have run its full 3 h and reported "STILL DOWN" against a live network; use `Resolve-DnsName` / `Test-NetConnection`; ssh's own "Could not resolve hostname" IS trustworthy (project_getent_false_negative.md).
271. **PowerShell `Get-Content`/`Set-Content` MANGLE UTF-8 in `.m` files** → PS 5.1 corrupts `µ`/`—`/`→` and MATLAB throws "Invalid text character" → build temp copies inside MATLAB (`fileread(...,'Encoding','UTF-8')` + `fopen(...,'w','n','UTF-8')`) (reference_matlab_local_verification.md).
272. **A PowerShell `type | ssh` pipe CORRUPTED an `authorized_keys` entry** → always write it via a remote `echo '<pubkey>'`, never a Windows pipe (project_igum_cluster.md).
273. **MATLAB identifiers cannot start with `_`** → invoke a temp script via `run('path/_tmp.m')`, never bare `_tmp`; and `-batch` pwd is the launching shell's cwd (repo root), so use absolute paths (reference_matlab_local_verification.md).
274. **MATLAB float round-trip breaks equality on radii** → `r_nm == 150` fails on `150e-9*1e9` → `round()` first (project_tm_scatterer_scan.md).
275. **`cd "$CLAUDE_SCRATCHPAD_DIR"` silently no-ops** → the env var is EMPTY in Bash here and `cd ""` succeeds, so downloads land in the repo root → use the literal scratchpad path (reference_air_trench_formulation_doc.md).
276. **lumapi build-smokes STALL in background shells** → run them in the foreground (project_loss_exploration_chain.md).
277. **Login-node container CAD (`fdtd-solutions`) SEGFAULTS even with license env** → read solved `.fsp` on compute jobs only; `updateportmodes`-in-layout probes also FAIL both locally and in the container ("Failed to evaluate code"), so there is no cheap pre-solve n_eff readout — the solve is the test (project_lumopt2_campaign_state.md, project_zoff_zmesh_knife_edge.md).
278. **Local lumapi on the IGUM login node needs the license env explicitly** → `ANSYSLMD_LICENSE_FILE=1055@132.68.48.51` (project_reference_air_trench_formulation_doc / trench_flare_apod round 2).
279. **Three local lumapi failures were VPN degradation, not licensing** → local lumapi works when the Technion VPN is up; the container-srun fallback needs `/opt/lumerical/v261/python/bin/python`, not `python3` (project_scatterer_greens_program.md).
280. **Compound shell forms are blocked by the permission classifier** → `cd ... && ENV=... bash deploy | grep | head` was blocked while the plain `ENV=... bash athena/deploy_athena.sh --lumopt2-design=...` went through; the block hit AFTER a cancel had already happened → order dispatch-critical steps so a block cannot strand a cancelled campaign; three install routes for a fixed cleaner (`find … -delete`, `cat > file` over ssh, `scp` of the script) were also blocked (the first two correctly, as CLAUDE.md §8 bypass forms) → ask rather than hunt for a fourth route (project_v2_width_gradient_plan.md).
281. **Env-var-prefixed ssh forms evade the permission rules** → always write remote commands host-first (`ssh evyatarrubin@athena.technion.ac.il "..."`), never `SSHHOST=... ssh "$SSHHOST" ...`; strip the Technion banner with `grep -vE "post-quantum|openssh|may need to be upgraded"` (CLAUDE.md §6).
282. **No LaTeX is installed on this machine** → use portable Tectonic (single exe, unzip into the scratchpad, `./tectonic.exe file.tex`) and Read the output PDF before delivering; the user explicitly rejects matplotlib PdfPages for formulation docs (reference_air_trench_formulation_doc.md).
283. **`checkcode` + a headless `exportgraphics` render is the MATLAB smoke test** → `matlab.exe -batch "msgs=checkcode('...','-string'); disp(msgs)"`, then a dialog-free copy with `set(0,'DefaultFigureVisible','off')` and `run(tmpPath)` (reference_matlab_local_verification.md).
284. **`--array-tasks=` accepts comma lists and is the right way to add ONE rung** → `--array-tasks=1` correctly dispatched only the new rung of a 2-row spec without re-running row 0; `--max-concurrent=` and `--gpu=` are also real flags (verified in the parser) (project_q3db_predictive_engine.md, project_v2_width_gradient_plan.md).
285. **Direct `scp` of a study's `*.mat` is faster than the deploy's `--results-no-fsp` menu** and avoids its known background hang (project_q3db_predictive_engine.md).

Coverage: this catalogue was harvested from lines 1–13053 of `all_mem.md` (the complete file).

## 8. Open threads and next steps, as recorded

### 8.1 Inverse design (the live programme)

- **Restart both lanes** with `wgp_reuse_travel_nm=40`, start cap 20, **ceiling 40 (not 60 —
  60 is where both lanes broke)**, slack 1.5e-3 now safe; reset d1's optstate cap 60→20 first;
  d1u additionally needs its λ-margin 0.2 knobs. Resume command:
  `SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_proj_d1`
  (or `_d1u`). **★ The `4d_1g` QOS REJECTS 300G — use 256G.** The three local fixes are gated
  but **UNDEPLOYED**.
- **N_FREE 25 → 60 is the top lever** (the light-cone ranker says +30 params buys ~160× model-
  leak reduction ≈ ~5× Q_i, saturating; the current 25 teeth give 16× ≈ 2.4×). Then **free the
  comb**, then the **TE lane**.
- **PARKED for the user**: the commit (engine fixes + gate + `BEST_D1_T9676` + docs are all
  uncommitted); code-compaction consolidation; deleting `scratch_s5vec.txt`; adaptive reuse
  depth k (log `cos∠(gW_fresh, gW_stored)` on every refresh and grow k with evidence); the
  width-band Pareto decision (σ +4.4% ↔ loss −31%); the PVA-vs-conformal Bloch-cell
  arbitration; Tier-2/3 deletions.
- **Literature-derived hardening not yet implemented** (`reference_method_lit_check_2026-08-27.md`
  — "if only three get done: #1, #2, #3"):
  1. an explicit **bandwidth / trust region on λ_pk drift** per iteration — the published
     high-Q recipe pairs a peak-recentred objective with `Re ω*(p) ∈ BW(ω₀)` as a HARD
     constraint, and our band-edge wrap guard is only half of that;
  2. **guard the IFT denominator** — `dλ_pk/dp = −(∂²T/∂λ∂p)/(∂²T/∂λ²)` has unbounded gain as
     the curvature → 0 (flat, merging, or critically-coupled peak); log `∂²T/∂λ²` every eval
     and clip the chain term;
  3. re-run the projection offline at ∂W/∂λ × 0.8 and × 1.2 and report how far the null-space
     direction rotates (it rotated ~50° in one smoke).
  Also: **mode volume V as a free independent cross-check** on `softW` and `C_field` (a smooth
  ratio of field integrals, no level set, no hyperparameter — if ∇V and ∇softW disagree in
  sign, one is wrong); the **`C_field` invariance test** (fit at ≥2 λ, ≥2 mesh sizes, ≥2
  monitor spans, ≥2 device lengths — constant ⇒ a legitimate normalization that should be
  DERIVED and hard-coded; drifting with mesh ⇒ a missing dV; with λ/position ⇒ a phase-
  reference bug; with device size ⇒ the tiling is wrong; **the −16.4° phase of
  0.4554 − 0.1336i is an intermediate phase, which is a bug signature, not a normalization**);
  **O(Q²) Hessian conditioning** — do not carry low-Q step sizes to high Q; **Fano asymmetry**
  invalidates "T-peak = mode resonance" and can give ∂T/∂λ = 0 a second root; **check forward-
  field sampling against the objective band's Nyquist rate** for the tiled field-monitor
  adjoint; **MMA/CCSA + epigraph** is a cheaper banked fallback than the augmented Lagrangian
  (the photonics default, and it needs no extra adjoint).
- **Ranked residual risks** (same file): (1) ∂W/∂λ is a fitted, path-dependent ±20% scalar
  *inside* the object that defines the feasible manifold; (2) no explicit λ-drift bandwidth
  constraint; (3) O(Q²) conditioning unpriced; (4) `C_field`'s phase unfalsified; (5) Fano
  asymmetry.
- **The structural argument for a real constraint** (the user's insight, still standing):
  penalties keep every iterate feasible, so **a design behind a temporarily-infeasible
  excursion is UNREACHABLE IN PRINCIPLE**; and the constraint is ONE scalar (width) enforced by
  TWO independent proxy walls on TWO parameter blocks, which structurally FORBIDS exactly the
  between-block σ-neutral trades. What ∇σ buys is the TANGENT direction, not permission to
  violate. **Platform directive (user, 2026-08-17): the goal is ONE automatic program** — every
  manual intervention in the corr-325 campaign traced to solving a CONSTRAINED problem with
  UNCONSTRAINED tools; any automation wrapper written before a real constraint function just
  freezes today's judgment calls into brittle thresholds.
- **Comb-count plan (4 steps, partly done)**: read the converged per-site radii as a marginal-
  value map; a count ladder at the surrogate (41/47/52 plus a few extensions); 3–4 comb-count
  rows at the production confirm; then a density-comb stage if structure appears. Measured so
  far: count is FLAT at 29/57/113, the knee is BELOW 29, **n = 29 halves the posts for free**;
  the close-out sweep was revised to n ∈ {7, 13, 21} (DROP 41).
- **Post-campaign protocol (unchanged)**: the same-N triplet readout (bare / bare+comb /
  optimized at N = 100, identical numerics) is the PRIMARY comparison; the **−3 dB confirm
  (2–4 plain forward sims at N ≈ 165–169 plus accurate mesh, NO optimizer)** is the SECONDARY
  translation onto the program benchmark scale; then the lock-target pitch re-trim, plus
  **fabrication ±5 nm bias rows**.
- **Highest-value remaining cheap experiment on record**: recover the VOID widths of the shift
  ladder ×0 / ×0.5 / ×1.5 on the BEST design (3–4 forward sims) — their T is valid but
  `fwhm_env` is `None`, so the clean constant-width shift test has never actually been run.
- Proposed but NOT done: seed a campaign FROM the hand-built see-saw (0.93836, in band) instead
  of a flat uniform start; the 2-forward pitch-locked `wcav` test (961 vs ~1100) that both
  prices the wall term and tests the 189 nm of headroom to the 1150 bound.

### 8.2 Comb / q3db programme

- After the 2026-09-12 delivery the programme is **COMPLETE for that request**.
- Open, user decision: **B3** = whether the apodized TM device (Q_i 217k bare / ~620k with
  trench, i.e. 4–10× the comb lock's 55k) enters the q3db benchmark; **B4** = TE.
- Follow-up not done: add a `tm_comb110_c325` family to `calibrate_q3db` (4 rows exist).
- The only comb run the model still justifies on the corr-400 device is an r-ladder at the
  cutoff (Λ 531 / 270° / N41 / d 1.8, r = 150, 180; model +0.029/+0.034 versus a saturation of
  ≤ +0.019; odds ~1/3).
- **B1 azimuthal 2-row WITHDRAWN** — the plan's hypothesis is not supported by the T data
  (b → ∞).

### 8.3 q3db predictive engine

- PARKED for the user: **TE > 1e5 hardening rows** (10–16 h each — "not short"); the **git
  push** (all repo work is committed on branch `add-claude-rules-skills`: e163c42, f7e06b5,
  6b676f2, 2f31d77, 0ce99bf, 1f6469c, c48cac8, 0ed931b plus a final commit — nothing
  uncommitted there, but the push itself needs permission); the **Phase-4 hybrid
  FDTDElement-style splice** (not triggered, B8 passed; every stored `.mat` already carries
  `S11`/`S21_complex` + `T_matrix` for it).
- Optional gap-fillers if wanted: second Q_c rows at corrugation 266/400 (N = 182), the c276
  saturation onset, and a TE > 1e5 pair at adequacy numerics (≤ 7 short runs).

### 8.4 Standing parked items across the whole programme

Accurate-mesh confirms (the §1.4 two-step) of: the [0, 270] scatterer winner, narrow-touch, the
trench family (W800 / W1050 / apod rows, the d-refine), the comb N = 41 / N = 47 candidates, the
air-comb cladding rows, both q3db operating points, and the TE periodic-comb candidate. A deeper
trench h-scan between 350 and 2000 nm (is the knee linear?) and a d-refine on W1050 / apod.
Archiving per the code-lifecycle rules when studies close (`runners/archive/`,
`matlab_plotting/studies/`). **Git commits and any deletion are permanently user-gated.**

### 8.5 Explicitly closed — do NOT re-propose

Distributed π-shift · step-envelope islands · inner-tooth shapes · wall-phase offset ·
anti-radiator asym-DW · hourglass as a device · external scatterers on width-optimized devices ·
cavity shape on top of rect-1050 · **the 2-pillar pair** · in-core holes (lattice, single, comb,
oxide and air, every phase / count / size) · in-core holes at equal width · shaped or curved
trench walls (the d(x) axis) · SiN lateral reflectors (strips, DBR, PhC) · metal lateral mirrors,
combs and corners · the retro-Bragg comb · the 2Λ superlattice · tall pillars · giant (r = 400)
scatterers · the 2D scatterer grid · comb + apodization stacking · full-length combs · a third
comb · fan combs · curves / chirps / clusters / rods in the comb family · FW-BIC (side-coupled) ·
symmetry-protected BIC · backward-Kerker · Huygens/Kerker scatterers · the single-cavity
supercavity · counterdiabatic as a special shape · moiré · x-asymmetry of the grating · the Si
handle-wafer feature · the tooth-response Green's matrix (user-rejected) · air holes as a device
(user-vetoed) · CMT width models inside lumopt2 · σ / participation-ratio width metrics · the
raw-line FWHM metric.


---

# Part 2 — Cluster operations manual

How to upload, run, monitor, fetch and stop work on the two Technion SLURM clusters.
Read from the scripts as they exist today; its internal section numbers are local to this part.


Audience: an AI assistant with repo access at `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes` and zero prior context. Everything below is read from the scripts as they exist today. Where a documented behavior contradicts the code, I say so under **AMBIGUITY**.

---

## 1. Hosts, accounts, paths, license

### 1.1 Athena (default cluster)

| Item | Value | Source |
|---|---|---|
| ssh target | `evyatarrubin@athena.technion.ac.il` | `athena/athena.conf` (`ATHENA_USER`, `ATHENA_HOST`) |
| ssh alias | `ssh athena` (ForwardAgent yes) | `c:\Users\evyat\.ssh\config` |
| Remote base | `/home/evyatarrubin/bragg_sim_athena` | `REMOTE_BASE` in `athena/athena.conf` |
| Remote layout | `project/ data/ results/layouts jobs/logs scripts/` (created by deploy) | `deploy_athena.sh` "Creating remote directories" |
| Job logs | `/home/evyatarrubin/bragg_sim_athena/jobs/logs/*.out` | `#SBATCH --output=logs/...` + `--chdir=${REMOTE_BASE}/jobs` |
| Results (remote) | `/home/evyatarrubin/bragg_sim_athena/results/<study>/results/*.mat` | `config.BASE_SAVE_DIR='/work/results'` + `RUN_NAME` |
| Results (local) | `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\` | `LOCAL_RESULTS_DIR` |
| Container | `$HOME/containers/lumerical-2026R1.sif` (filename fixed; ~6 job scripts hardcode it) | all `athena/jobs/*.sh` |
| `LUM_HOME` inside container | `/opt/lumerical/v261` | job scripts |
| Engine binary | `/opt/lumerical/v261/bin/fdtd-engine-ompi-lcl` | `run_fsp_gpu*.sh` |
| Container python | `/opt/lumerical/v261/python/bin/python` | `run_python_*.sh` |
| lumapi | `/opt/lumerical/v261/api/python/lumapi.py` | `athena/scripts/athena_run.py` |
| CPUs per task | `N_CPUS=8` | `athena.conf` |
| Default partitions | `h200-shared,a100-public,rtx6k-shared,l40s-public,l40s-shared` | `ARRAY_PARTITIONS` |
| Default QOS | `ARRAY_QOS=24h_1g`; aggregator `AGG_QOS=4h_0g` on `AGG_PARTITION=l40s-public` | `athena.conf` |
| Default walltime | `ARRAY_TIME=23:30:00` | `athena.conf` |
| Concurrency throttle | `MAX_CONCURRENT=8` (array `%K` suffix) | `athena.conf` |
| Mail | `evyatar10.rubin@gmail.com` (BEGIN/END/FAIL, ARRAY_TASKS for arrays) | job scripts |

### 1.2 IGUM (second, coexisting cluster)

| Item | Value | Source |
|---|---|---|
| ssh target | `evyatarrubin@132.68.58.101` (igum-login1; FQDN does **not** resolve off-cluster) | `igum/igum.conf` (`IGUM_USER`, `IGUM_HOST`) |
| ssh alias | `ssh igum` | `~/.ssh/config` |
| Access | Technion **VPN required** | `igum/README.md` |
| Remote base | `/home/evyatarrubin/research/bragg_sim_igum` (via `~/research` → `/research/amir.r/evyatarrubin`) | `igum.conf` |
| Results (local) | `...\results_from_igum\` | `LOCAL_RESULTS_DIR` in `deploy_igum.sh` |
| Lumerical (native, no container) | `/home/evyatarrubin/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261` | `LUM_HOME` in all 6 `igum/jobs/*.sh` |
| Fallback tree (kept) | `/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261` | `igum/README.md` §4 |
| Engine binary | `${LUM_HOME}/bin/fdtd-engine` (plain — the `-ompi-lcl` variant fails, no `libmpi.so.40`) | `igum/jobs/run_fsp_gpu*.sh` |
| Account (mandatory) | `SLURM_ACCOUNT="acct-lumerical"` | `igum.conf` |
| QOS / partition | `ARRAY_QOS=qos-preempt` / `ARRAY_PARTITIONS=part-preempt`; agg `qos-lumerical` / `part-lumerical` | `igum.conf` |
| GPU request | `GPU_REQ="--gres=gpu:1"` | `igum.conf` |
| Concurrency | `MAX_CONCURRENT=4` (seats shared with Athena) | `igum.conf` |
| scilibs | `${REMOTE_BASE}/scilibs` on `LD_LIBRARY_PATH` (part-preempt nodes are bare) | `igum/jobs/run_python_array.sh` |

**AMBIGUITY:** `igum/igum.conf` sets `SLURM_ACCOUNT="acct-lumerical"`, but `igum/README.md` §3 says the verified working recipe is `--account=acct-preempt --partition=part-preempt --qos=qos-preempt`. The conf pairs `acct-lumerical` with `part-preempt`/`qos-preempt`. Confirm with `sshare -U` / `scontrol show assoc_mgr` before a first IGUM dispatch; a mismatch yields "Invalid account or account/partition combination".

### 1.3 License (one FlexLM server, shared by both clusters)

- Server IP `132.68.48.51`, hostname `lumerical-lm.ece.technion.ac.il`.
- Ports: **1055** (lmgrd) and **2325** (Ansys interconnect).
- Conf vars (identical in both confs, names kept `ATHENA_*` on purpose):
  `ATHENA_LICENSE="1055@132.68.48.51"`, `ATHENA_INTERCONNECT="2325@132.68.48.51"`.
- Exported into the job env by every job script as:
  `ANSYSLMD_LICENSE_FILE='1055@132.68.48.51'`, `ANSYSLI_SERVERS='2325@132.68.48.51'`, plus `ANSYS_APIP_DISABLE=1`.
- Athena only: the hostname is **not in DNS**, so each job builds/binds `$HOME/hosts_lum` (copy of `/etc/hosts` + the line `132.68.48.51 lumerical-lm.ece.technion.ac.il lumerical-lm`) onto `/etc/hosts` in the container. IGUM resolves natively — no hosts hack.
- lmutil paths: Athena container `/ansys_inc/v261/licensingclient/linx64/lmutil`; IGUM native `~/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/licensingclient/linx64/lmutil`.
- Measured pool facts (`igum/README.md` §5): 50 issued `lum_fdtd_solve` seats, effective ~42; **one GPU solve ≈ 7 seats ⇒ hard ceiling ~6 concurrent solves across both clusters**.

---

## 2. Complete flag surface of `athena/deploy_athena.sh`

The parser is a single `for arg in "$@"` case block (lines 64–103). **Any unrecognized flag aborts before submission** (added 2026-08-16 after two invented flags were silently ignored). `deploy_igum.sh` has the same parser except `--gpu=` (error) — see §6.

### 2.1 Mode selection

| Flag | Effect |
|---|---|
| `--option1` | `OPTION=1` — engine mode: generate `.fsp` locally, upload, run `fdtd-engine` on GPU (no Python on the node) |
| `--option2` | `OPTION=2` — lumapi Python pipeline, one sequential job (`jobs/run_python_gpu.sh`) |
| `--option3` | `OPTION=3` — lumapi pipeline as a SLURM array (`jobs/run_python_array.sh`); requires `--sweep=` or `--spec=` |
| `--run=<module>` | Sets `RUN_SCRIPT`; bare module name from `runners/single/`, `runners/tm/`, or `convergence_testing/` |
| `--preset=<p>` | Option 1 layout preset: `single` \| `sweep_shift` \| `sweep_inner_size` |
| `--fsp=<name>` | Option 1 with an explicit already-present `.fsp` name (skips local generation) |
| `--sweep=<kind>` | Sets `SWEEP_KIND` only (does **not** set OPTION=3 — pass `--option3` too) |
| `--spec=<module>` | `OPTION=3`, `SWEEP_KIND=spec`, `SPEC_MODULE=<module>` — the normal sweep form |
| `--cards=<module>` | `OPTION=3`, `SWEEP_KIND=cards` — experiment-card lists |
| `--inverse-design=<module>` | `OPTION=3`, `SWEEP_KIND=inverse_design` |
| `--gradient-free-design=<module>` | `OPTION=3`, `SWEEP_KIND=gradient_free_design` |
| `--lumerical-native=<module>` | `OPTION=3`, `SWEEP_KIND=lumerical_native_optimization` |
| `--fd-gradient-design=<module>` | `OPTION=3`, `SWEEP_KIND=fd_gradient_design` |
| `--lumopt2-design=<module>` | `OPTION=3`, `SWEEP_KIND=lumopt2_design` |
| `--pol-array` | Option 2 only: submits as `--array=0-1%2` (task 0 = TE, task 1 = TM) |

### 2.2 Upload / info / download (each exits without submitting)

| Flag | Effect |
|---|---|
| `--upload-only` | rsync everything, `chmod +x` job scripts, then exit. **This is the only correct "push code, don't run" flag.** |
| `--status` | `squeue -u evyatarrubin -o '%.10i %.12j %.8T %.10M %.6D %R'` + `tail -40` of the newest `jobs/logs/*.out`; exit |
| `--gpu-status` | Per-partition free-GPU table (sinfo/scontrol/squeue over `h200-shared a100-public l40s-public l40s-shared rtx6k-shared`); exit |
| `--license-probe` | Runs `lmutil lmstat -a -c 1055@132.68.48.51 \| grep -E 'Users of (lum_fdtd\|fdtd_)'` on the cluster; exit |
| `--results` | Prompts: 1) data only 2) full incl. `.fsp` 3) specific files |
| `--results-no-fsp` | `ssh ... "tar --exclude='*.fsp' -czf - -C '<REMOTE>/results' ."` piped into local `tar -xzf -` |
| `--results-full` | `scp -r <SSH>:<REMOTE>/results/. results_from_athena/` (heavy) |
| `--results-files` | 3-step interactive picker: category → run folder → files (`1,3,5` \| `1-3,7` \| `all`) |
| `--results-files=p1,p2,...` | Non-interactive; paths relative to `results/`, fetched in one tar stream |

### 2.3 Resource / scheduling overrides

| Flag | Effect |
|---|---|
| `--gpu=<type>` | Replaces `ARRAY_PARTITIONS` with ONE partition. Accepted: `h200`\|`h200-shared` → `h200-shared`; `a100`\|`a100-public` → `a100-public`; `l40s`\|`l40s-public` → `l40s-public`; `l40s-shared`; `rtx6k`\|`rtx`\|`rtx6k-shared` → `rtx6k-shared`. Anything else = hard error. Applies to all options. |
| `--array-tasks=<spec>` | Overrides the array range (default `0-<N-1>`); `%K` still appended. Use `--array-tasks=0` for a one-task smoke, `--array-tasks=1-100` for QOS chunking, or a dead-index list for a resubmit. |
| `--max-concurrent=<K>` | Overrides `MAX_CONCURRENT` (the `%K` throttle) |
| `--after=<jobid>` | Adds/extends `--dependency=afterok:<jobid>` on the option-3 array |
| `--qos=<name>` | Overrides `ARRAY_QOS` — **option-3 path only** (read inside the array branch) |
| `--time=<HH:MM:SS>` | Overrides `ARRAY_TIME` — **option-3 path only** |
| `--keep-h5` | Sets `KEEP_H5=1` → `cfg.run.cleanup_lumerical_data = False` (keeps `.h5` scratch; quota risk) |

### 2.4 Environment variables the deploy honors (not flags)

- `SBATCH_MEM=<size>` → adds `--mem=<size>`. Honored in the **option-2** and **option-3** branches only (not option 1). Defaults come from the job scripts: 64G (single/engine), 128G (python array).
- `LUMOPT2_QOS` / `LUMOPT2_TIME` → override QOS/walltime, **only when `SWEEP_KIND=lumopt2_design`**. Example from the script's own comment:
  `LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 SBATCH_MEM=160G bash athena/deploy_athena.sh --lumopt2-design=<module>`
- `PRELIM_TIME` → walltime for an auto-submitted prelim job (default `00:10:00`).
- `REQUIRE_GPU` (default 1 for arrays/option2 export), `CONV_POL` (default `TE`), and the TM knob set forwarded verbatim on option 2: `TM_PITCH_NM` (default 500), `TM_WIDE_PITCH_NM`, `TM_SIM_TIME_PS`, `TM_WIDE_SINGLE_CORR_NM`, `TM_HEIGHT_NM` (350), `TM_SCAN_CENTER_NM`, `TM_SCAN_WIDTH_NM`, `TM_SCAN_NPTS`, `TM_WIDE_CENTER_NM`, `TM_WIDE_WIDTH_NM`, `TM_WIDE_NPTS`, `TM_WIDE_SEEDS_NM`, `TM_MESH` (`optimization`), `TM_FARFIELD` (0), `TM_RECORD_2D` (0), `SPAN_MULT`, `TM_CONST_MODE` (`sampled`), `NTFY_TOPIC`.
- **`ARRAY_TIME=...` as an env var is silently IGNORED** — `athena.conf` plain-assigns it after being sourced. Use `--time=` (option 3) or `LUMOPT2_TIME`.

### 2.5 Interactive menus (no flags → prompts)

**Top level:** `1)` FSP job (engine) — `2)` Python job (lumapi pipeline).

**Option-1 preset menu:** `1) single`, `2) sweep_shift`, `3) sweep_inner_size`.

**Option-2 pipeline sub-menu** (the "deploy menu of studies"):

| # | Label | Routing |
|---|---|---|
| 1 | Single (one node, sequential — `runners/single/`) | `OPTION=2`, picker over `runners/single/*.py` |
| 2 | Sweep (parallel SLURM array — `runners/sweeps/`) | `OPTION=3`, `SWEEP_KIND=spec` |
| 3 | Inverse design (lumopt adjoint — `runners/inverse_design/`) | `OPTION=3`, `SWEEP_KIND=inverse_design` |
| 4 | Gradient-free design (Lumerical PSO — `runners/gradient_free_design/`) | `OPTION=3`, `SWEEP_KIND=gradient_free_design` |
| 5 | Lumerical-native opt (`addsweep('Optimization')`) | `OPTION=3`, `SWEEP_KIND=lumerical_native_optimization` |
| 6 | FD-gradient design (scipy L-BFGS-B + central diff) | `OPTION=3`, `SWEEP_KIND=fd_gradient_design` |
| 7 | Convergence (`convergence_testing/`, incl. mesh_conv X/YZ) | array if the file has `PHASES`+`KIND_PREFIX`, else sequential |
| 8 | TM studies (`runners/tm/`) | `OPTION=2`, picker over `runners/tm/*.py` |

Note there is **no menu entry for `lumopt2_design`** — it is reachable only via `--lumopt2-design=<module>`.

**Discovery rules used by the pickers** (these are the "deploy-menu contract"):
- `runners/single/`, `runners/tm/`, `convergence_testing/`: file must match `^(def[[:space:]]+run[[:space:]]*\(|run[[:space:]]*=)`; skipped if name starts with `_`, is `__init__.py`, or the file has `^IS_HELPER[[:space:]]*=[[:space:]]*True`.
- Sweep/optimization dirs: file must contain the literal text `SPEC =` **anywhere** (unanchored `grep -rl` — a comment or docstring counts); the engine file (`sweep_spec.py`, `inverse_design.py`, …) and `test_geometry.py` are excluded by filename; `^IS_PRELIM = True` hides a sweep module.
- Convergence: a file with `PHASES = [...]` and `KIND_PREFIX = "..."` expands into one array entry per phase (`SWEEP_KIND=<prefix>_<phase_lower>`); otherwise one sequential entry.

### 2.6 Known flag-surface defects

- **`--watch` and `--watch-only` do not exist in the parser.** They are advertised in the script's own usage header (lines 18–19) and offered by `.vscode/tasks.json` ("Watch job status (no deploy)"), but the case block has no branch → they hit `*)` and abort with `ERROR: unknown flag`. Use `--status` instead. Same on IGUM.
- **`--option1 --preset=...` is broken**: it calls `${LOCAL_PROJECT}/hpc/scripts/local_save_fsp.py`, and neither `hpc/` nor that file exists in the repo. Option 1 is only usable via `--fsp=<name>` against an already-uploaded stem. Treat the engine path as untested/legacy.
- `--qos=` / `--time=` are parsed globally but only *read* in the option-3 branch; on `--option1`/`--option2` they are silently ineffective.

---

## 3. Step-by-step: dispatching a study

### 3.0 Pre-dispatch gates (from `dispatch-study` + `athena-preflight`)

1. State in one line what the run decides and why stored results can't answer it. Enumerate points that already exist in `results_from_athena/` / `results_from_igum/` and cut them.
2. **Ask which cluster** ("Athena or IGUM?") unless the user already named one.
3. Echo the *built* config (pitch, n_core, N periods, which monitors are on) from the SPEC/runner file — not the intent.
4. Preflight, three checks:
   ```bash
   bash athena/deploy_athena.sh --license-probe
   ssh evyatarrubin@athena.technion.ac.il "for p in 1055 2325; do timeout 8 bash -c \"cat </dev/null >/dev/tcp/132.68.48.51/\$p\" 2>/dev/null && echo \"port \$p OPEN\" || echo \"port \$p CLOSED\"; done"
   bash athena/deploy_athena.sh --status
   ssh evyatarrubin@athena.technion.ac.il "du -sh ~ 2>/dev/null; quota -s 2>/dev/null || true"
   ```
   Seat count (reliable vantage = IGUM):
   ```bash
   ssh igum '$HOME/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve' | grep "Users of lum_fdtd_solve"
   ```
   Bands: ≥35/50 in use = HIGH (hold fan-outs); ≥45/50 = CRITICAL (no dispatch). Athena `lmstat` returning `-96` is a documented **false negative** — do not block on it; ports open by IP is the real signal.
5. Count pending array tasks with `squeue -r` (plain `squeue` collapses a pending array to one line and undercounts). QOS `24h_1g` caps **100 submitted / 4 running** tasks per user → chunk with `--array-tasks=`.
6. Smoke first if the change touches geometry, a new builder, gradients, or source/BC setup. For the four optimization families that means dispatching that family's `smoke_test.py` before `optimize_transmission.py`.

### 3.1 Create/edit the runner

Copy the closest sibling; do not scaffold new infrastructure.

- Sweep: `runners/sweeps/<name>.py` with a top-level `SPEC = SweepSpec(...)` (optionally `BASE`, `PRELIM_SPEC`, `PRELIM_BASE`, `PRELIM_RUN_SCRIPT`, `LOCKED_LAMBDA_FILE`).
- Single: `runners/single/<name>.py` with a top-level `run` callable; TM work → `runners/tm/`.
- Optimization: copy `optimize_transmission.py` / `smoke_test.py` in the family dir; base config from `runners/optimization_common.py::make_optimization_base(n_periods)`.
- Optional hooks honored by `athena_run.py`: `build_cfg(cfg) -> cfg` (applies overrides on the cluster — a runner's `__main__` block is **never** executed there) and `STUDY_DIR_NAME` (forces the results folder, so a family of runners shares one study dir).

### 3.2 Dispatch commands (exact)

```bash
# Single run
bash athena/deploy_athena.sh --option2 --run=run_simulation

# TE/TM parallel pair (2-task array, task 0 = TE, task 1 = TM)
bash athena/deploy_athena.sh --option2 --run=run_tm_vs_te --pol-array

# Sweep array
bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.number_of_periods
bash athena/deploy_athena.sh --spec=runners.sweeps.number_of_periods      # --spec implies --option3

# One-task smoke of a sweep
bash athena/deploy_athena.sh --spec=runners.sweeps.<study> --array-tasks=0 --max-concurrent=1

# Experiment cards
bash athena/deploy_athena.sh --cards=runners.experiment_comparison.it11_devices

# Optimization families
bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.smoke_test
bash athena/deploy_athena.sh --inverse-design=runners.inverse_design.optimize_transmission
bash athena/deploy_athena.sh --fd-gradient-design=<module>
bash athena/deploy_athena.sh --lumerical-native=<module>

# lumopt2 campaign (long stateful driver)
LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 SBATCH_MEM=160G \
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.<module>

# Chain a stage behind an in-flight job
bash athena/deploy_athena.sh --spec=runners.sweeps.<stage2> --after=146639

# Push code with no run
bash athena/deploy_athena.sh --upload-only
```

### 3.3 What the deploy actually does, in order

1. `source athena/athena.conf`; resolve `LOCAL_NEFF` via `python -c "import config; print(config.NEFF_DATA_PATH)"` and `cygpath -u` it.
2. `--gpu=` partition pin (if given).
3. Interactive pickers (if needed).
4. Short-circuit actions: `--status`, `--gpu-status`, `--license-probe`, `--results*` all exit here.
5. `ssh <SSH> "mkdir -p ${REMOTE_BASE}/{project,data,results/layouts,jobs/logs,scripts}"`.
6. Uploads:
   ```
   rsync -av --itemize-changes <proj>/*.py               → REMOTE/project/
   rsync -av --delete --itemize-changes <proj>/runners/   → REMOTE/project/runners/
   rsync -av --delete --itemize-changes <proj>/convergence_testing/ → REMOTE/project/convergence_testing/
   rsync -av --itemize-changes <LOCAL_NEFF>              → REMOTE/data/FDE_sweep_results.mat
   rsync -av --itemize-changes athena/jobs/*.sh          → REMOTE/jobs/
   rsync -av --itemize-changes athena/scripts/*.py       → REMOTE/scripts/
   ssh <SSH> "chmod +x ${REMOTE_BASE}/jobs/*.sh"
   ```
   `--delete` on `runners/` and `convergence_testing/` means a locally deleted/renamed file disappears from the server copy. **Verify the itemized output actually lists the files you edited** — restrictive perms have silently made rsync skip root `*.py` (`--inplace` is the known fix).
7. `--upload-only` exits here.
8. Option 3 only: run `athena/scripts/build_sweep_list.py --kind <kind> [--module <mod>] --output results_from_athena/_sweep_list.txt`, capture `$?` **before** echoing (a past bug submitted job 133394 on a failed build), parse `SWEEP_META:` lines into `SWEEP_PARAM`, `SWEEP_FIXED_DYZ_NM`, `SWEEP_FIXED_CELLS`, `SWEEP_SPEC_MODULE`, `prelim_run_script`, `has_prelim_spec`, `locked_lambda_file`.
9. Upload the **per-study** list: `scp _sweep_list.txt <SSH>:${REMOTE_BASE}/data/sweep_list_<STUDY_TAG>.txt`, where `STUDY_TAG=${SPEC_MODULE##*.}` (falls back to `default` for kinds with no module, e.g. `mesh_conv_x`). The task-side env is `SWEEP_LIST=/work/data/sweep_list_<STUDY_TAG>.txt`. This makes one study's deploy structurally unable to invalidate another study's pending/REQUEUEd task indices.
10. Optional prelim chain: if the module declares `PRELIM_SPEC`, a prelim array is submitted first (`SWEEP_IS_PRELIM=1`, list `data/prelim_sweep_list.txt`) and the main array gets `--dependency=afterok:<prelim>`. If instead it declares `PRELIM_RUN_SCRIPT`, a single `run_python_gpu.sh` job is submitted the same way. `LOCKED_LAMBDA_FILE` (may contain `{idx}`) is forwarded to both.
11. `sbatch` (see §5 for the exact assembled command) → prints `Submitted: Submitted batch job <ID>`.
12. Post-submit auto-chains: `mesh_conv_x`/`mesh_conv_yz` queue `jobs/run_mesh_aggregate.sh` with `--dependency=afterok`; `runners.sweeps.tm_te_shift` and `runners.sweeps.tm_shift_p518` seed baselines and queue `jobs/run_shift_summary.sh`.

### 3.4 Report the job ID

Every dispatch turn ends with the submitted job/array ID and the task count, or a prominent "NOT dispatched because Y". The deploy prints:
```
Submitted: Submitted batch job <ID>
Live log on Athena cluster:
  ssh evyatarrubin@athena.technion.ac.il tail -f /home/evyatarrubin/bragg_sim_athena/jobs/logs/lum_*<ID>*.out
```
Verify the walltime actually granted: `sacct -j <ID> --format=JobID,TimeLimit,State`.

---

## 4. Checking status and fetching results

### 4.1 Status

```bash
bash athena/deploy_athena.sh --status
```
or one filtered round-trip (preferred for agents; strip the Technion banner):
```bash
ssh evyatarrubin@athena.technion.ac.il "squeue -u evyatarrubin -o '%.12i %.30j %.8T %.10M %R'" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"

ssh evyatarrubin@athena.technion.ac.il "cd ~/bragg_sim_athena/jobs/logs && ls -t lum_*.out 2>/dev/null | head -5 && echo '--- newest ---' && tail -30 \$(ls -t lum_*.out | head -1)" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"

ssh evyatarrubin@athena.technion.ac.il "ls -lt ~/bragg_sim_athena/results/<study>/results/result_*.mat 2>/dev/null | head -15" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"
```
Rules: always host-first (`ssh user@host "..."`) — never `SSHHOST=... ssh "$SSHHOST"`, which evades the permission-rule matching including the `scancel` ask-guard. Count array tasks with `squeue -r`. Automated polling budget: **≤3–6 ssh/hour per cluster, ONE connection per poll** (fold queue + log + lmstat into the same ssh). On any auth refusal, stop all automated contact ≥45 min, then one probe — never retry-loop.

Log-name patterns: `lum_fdtd_gpu-%j.out`, `lum_sweep_gpu-%A_%a.out`, `lum_pipeline_gpu-%j.out`, `lum_array-%A_%a.out`, `mesh_aggregate-%j.out`, `shift_summary-%j.out`.

### 4.2 Fetch

```bash
bash athena/deploy_athena.sh --results-no-fsp     # tar stream, excludes *.fsp  → results_from_athena/
bash athena/deploy_athena.sh --results-full       # scp -r everything incl. .fsp
bash athena/deploy_athena.sh --results-files=tm_te_shift/results/result_X.mat,tm_te_shift/results/shift_summary.csv
```
Targeted scp for one study:
```bash
mkdir -p results_from_athena/<study>/results
scp "evyatarrubin@athena.technion.ac.il:~/bragg_sim_athena/results/<study>/results/result_*.mat" results_from_athena/<study>/results/
```
IGUM equivalents: `bash igum/deploy_igum.sh --results-no-fsp` → `results_from_igum/`.

**Server-side reduction is mandatory for field data.** The link runs ~0.5–1 MB/s and a full field-profile `.mat` is ~650 MB while a figure needs one plane at one λ (~1 MB). Login-node `python3` has numpy/scipy — slice on the server, download the slice. Pull small state files (`*_evals.jsonl`, `*_optstate.json`, csv) on **every** milestone check; never let one cluster hold the only copy of a campaign's incremental log.

Repo helper for card batches: `runners/archive/experiment_comparison/pull_by_subname.py` (atomic `.downloading` temp + 300 MB size floor; never deletes or overwrites). Scratch cleaner for lumopt2 `.h5`: `athena/h5_clean_once.sh` (run from cron; keeps the newest four `*_output.h5` per live study, drops all scratch in study dirs idle >24 h; touches only `*_output.h5`).

### 4.3 Stopping runs

Confirm-first, never blanket. Resolve the ID from `squeue`, state it back, then:
```bash
ssh evyatarrubin@athena.technion.ac.il "scancel <ID> [<ID2> ...]"
```
`scancel` is on the permission **ask** list — that prompt *is* the confirmation. Re-check `squeue` afterwards. For arrays: the bare array ID kills all tasks; `<ID>_<task>` kills one.

---

## 5. Job scripts (`athena/jobs/*.sh`)

All five GPU/CPU scripts share the same container plumbing. Common invocation shape:

```
apptainer exec --nv \
    --bind <dirs...> --bind "${HOME}/hosts_lum:/etc/hosts" --bind "${HOME}/scilibs:/scilibs" \
    --pwd /work "$HOME/containers/lumerical-2026R1.sif" bash -c "<inner script>"
```

Inner-script env, in order, identical across scripts:
- `LANG=C LC_ALL=C`
- **Strip** `/usr/local/cuda/compat*` from `LD_LIBRARY_PATH` (the container's CUDA-12.2 forward-compat shim is older than Athena's R570+ host driver; leaving it causes `cudaGetDeviceCount Failed: unsupported display driver / cuda driver combination`). `--nv` then injects the host libcuda via `/.singularity.d/libs`.
- `LUMERICAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"` — because `fdtd-solutions` resets `LD_LIBRARY_PATH` to `$FDTD_LD_LIBRARY_PATH:$LUMERICAL_LD_LIBRARY_PATH` at relaunch, which would otherwise wipe everything.
- `LD_PRELOAD="/opt/lumerical/v261/lib/libtbbmalloc.so.2:/opt/lumerical/v261/lib/libtbbmalloc_proxy.so.2"` (prevents `free(): invalid pointer`).
- `ANSYSLMD_LICENSE_FILE`, `ANSYSLI_SERVERS`, `ANSYS_APIP_DISABLE=1`.
- Fabric guards: `RDMAV_FORK_SAFE=1 FI_EFA_FORK_SAFE=1 FI_PROVIDER='^efa' OMPI_MCA_btl=self,tcp OMPI_MCA_mtl='^ofi' UCX_TLS=self,sm,tcp UCX_NET_DEVICES=lo`.
- Append `:/scilibs` (libgfortran, libquadmath for scipy/numpy) as a **suffix** so `--nv`-injected driver libs still win.
- Python paths: `Xvfb :99 -screen 0 1024x768x24 -nolisten tcp` managed manually (`xvfb-run`'s shutdown returns non-zero and flipped SLURM to FAILED despite saved results), `DISPLAY=:99`, `sleep 1`, then the python call; exit code propagated explicitly (`echo "[wrapper] python exit code: $PY_RC"; exit $PY_RC`).
- No NVML trampoline (the R470 shim died with the DGX cluster; mounting it corrupts CUDA init — job 76907).

| Script | SBATCH defaults | Payload |
|---|---|---|
| `run_fsp_gpu.sh` | `--job-name=lum_fdtd_gpu --nodes=1 --gpus=1 --cpus-per-task=8 --mem=64G`, log `logs/lum_fdtd_gpu-%j.out` | `FSP_FILE` env → `RESULTS_ROOT=/home/evyatarrubin/bragg_sim_athena/results`, `FSP_DIR=$RESULTS_ROOT/<stem>` bound to `/work/layouts`; runs `fdtd-engine-ompi-lcl -t $SLURM_CPUS_PER_TASK -logall -use-gpu-resources /work/layouts/$FSP_FILE` in background with a **120 s nvidia-smi watchdog** (kills the engine if GPU mem <200 MiB) when `REQUIRE_GPU=1` (default). Also a license pre-flight: `lmutil lmstat -a` must contain `Users of lum_fdtd_solve`, else exit 2. |
| `run_fsp_gpu_array.sh` | `--job-name=lum_sweep_gpu --gpus=1 --cpus-per-task=8 --mem=64G`, log `logs/lum_sweep_gpu-%A_%a.out` | Reads stem from line `SLURM_ARRAY_TASK_ID+1` of `$RESULTS_ROOT/fsp_list.txt`; same engine + watchdog + license gate |
| `run_python_gpu.sh` | `--job-name=lum_pipeline_gpu --gpus=1 --cpus-per-task=8 --mem=64G`, log `logs/lum_pipeline_gpu-%j.out`; `REQUIRE_GPU` default **0** here | Binds `project→/work/project`, `scripts→/work/scripts`, `data→/work/data`, `results→/work/results`, `logs→/work/logs`; runs `/opt/lumerical/v261/python/bin/python /work/scripts/athena_run.py` |
| `run_python_array.sh` | `--job-name=lum_pipeline_array --gpus=1 --cpus-per-task=8 --mem=128G`, `--mail-type=END,FAIL,ARRAY_TASKS`, log `logs/lum_array-%A_%a.out`; `REQUIRE_GPU` default **1** | Same binds; exports `SWEEP_KIND SWEEP_LIST SWEEP_INDEX=$SLURM_ARRAY_TASK_ID SWEEP_PARAM SWEEP_FIXED_DZ SWEEP_FIXED_CELLS SWEEP_SPEC_MODULE REQUIRE_GPU KEEP_H5 LOCKED_LAMBDA_FILE`; runs `python -u /work/scripts/athena_run_one.py` |
| `run_mesh_aggregate.sh` | `--job-name=mesh_aggregate --cpus-per-task=2 --mem=4G` (CPU-only, no `--nv`) | Requires `PHASE=X\|YZ`; runs `run_mesh_convergence.py --aggregate $PHASE` via `runpy` with `config.BASE_SAVE_DIR='/work/results'` patched in |
| `run_shift_summary.sh` | `--job-name=shift_summary --cpus-per-task=2 --mem=4G` | `SUMMARY_DIR` (default `/work/results/tm_te_shift/results`), `X_AXIS` (default `absolute`, `relative` = % of half-pitch); runs `runners/sweeps/plot_tm_te_shift.py --results-dir … --x …` |

Walltime/QOS/partition/mem/array/dependency are **not** in the scripts — the deploy supplies them on the `sbatch` command line. The assembled option-3 submission is:

```
cd ${REMOTE_BASE} && sbatch ${MEM_OPT} \
  --array=${ARRAY_RANGE}%${K} ${DEP_FLAG} \
  --gpus=1 --cpus-per-task=8 \
  --time=${ARRAY_TIME:-00:50:00} \
  --qos=${ARRAY_QOS} \
  --partition=${ARRAY_PARTITIONS} \
  --export=ALL,SWEEP_KIND=...,SWEEP_LIST=/work/data/sweep_list_<tag>.txt,SWEEP_PARAM=...,SWEEP_FIXED_DYZ_NM=...,SWEEP_FIXED_CELLS=...,SWEEP_SPEC_MODULE=...,ATHENA_LICENSE=...,ATHENA_INTERCONNECT=...,REQUIRE_GPU=1,KEEP_H5=0,CONV_POL=TE,NTFY_TOPIC=[,LOCKED_LAMBDA_FILE=...] \
  --chdir=${REMOTE_BASE}/jobs \
  jobs/run_python_array.sh
```

### 5.1 Cluster-side python entry points

`athena/scripts/athena_run.py` (sequential, OPTION=2): inserts `/work/project` on `sys.path`; patches `config.BASE_SAVE_DIR='/work/results'`, `NEFF_DATA_PATH='/work/data/FDE_sweep_results.mat'`, `USE_GPU=True`, `LUMAPI_PATH='/opt/lumerical/v261/api/python/lumapi.py'`; builds `cfg = SimulationConfig()` with `mesh.simulation_mode="optimization"` and `grating.cavity_neg_detuning_nm=5.76`; `cfg.run.cleanup_lumerical_data = (KEEP_H5 != "1")`; auto-discovers runners in `_AUTO_DIRS = [runners.single, runners.tm, convergence_testing]`; resolves `_ALIAS_SCRIPTS` (`compare_3d_field_prelim`, `run_mesh_convergence`, legacy `single_sim`, `simple_bragg`); honors `STUDY_DIR_NAME` and `build_cfg(cfg)`; patches `lumapi.FDTD.__init__` to call `setresource("FDTD",1,"device type","GPU")` and, with `REQUIRE_GPU=1`, `sys.exit(2)` if that fails or reads back non-GPU; on success `os._exit(0)` (lumapi's shutdown otherwise returns non-zero and marks the job FAILED).

`athena/scripts/athena_run_one.py` (one array task): same patching, plus `SWEEP_KIND` validated against
`{shift, inner_size, generic, mesh_conv_x, mesh_conv_yz, shutoff_conv_shutoff, spec, cards, inverse_design, gradient_free_design, lumopt2_design, lumerical_native_optimization, fd_gradient_design}`, `SWEEP_INDEX` (= `SLURM_ARRAY_TASK_ID`) indexed into `SWEEP_LIST`. `kind=spec` imports `SWEEP_SPEC_MODULE`, expands `SPEC.expand(base=BASE)` (or `PRELIM_SPEC`/`PRELIM_BASE` when `SWEEP_IS_PRELIM=1` or `IS_PRELIM=True`), takes `configs[idx]`, and applies/writes the `LOCKED_LAMBDA_FILE` sidecar (`{idx}` expanded; prelim writes `lambda_res_nm`/`lambda_res_m`, main reads and overwrites `cfg.spectral.center_wavelength_m`).

`athena/scripts/build_sweep_list.py`: `--kind {mesh_conv_x,mesh_conv_yz,shutoff_conv_shutoff,spec,cards,inverse_design,gradient_free_design,lumerical_native_optimization,fd_gradient_design,lumopt2_design}`, `--output`, `--module`, `--prelim`, `--cards-manifest`. Line counts come from `SPEC.expand()`, `CARDS`, `spec.get_starts()`, or `N_TASKS` (lumopt2, default 1). Emits `SWEEP_META: key=value` lines the deploy greps.

### 5.2 Container recipe (`container/lumerical.def`)

Not the routine upgrade path — use the `update-container` skill (on-Athena sandbox surgery, ~1 h; the 5 GB `.sif` never crosses the VPN). The recipe: `Bootstrap: docker` / `From: nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04`; `%files` copies a pre-staged tree (`/home/evyatar10/ansys_incS_R13/v261/Lumerical`) to `/opt/lumerical/v261` plus its `licensingclient` to `/ansys_inc/v261/licensingclient` (the engine hardcodes that path for `ansyscl`); `%post` installs the xcb/GL/Xvfb dependency set and chmods `bin/fdtd-*`; `%environment` sets `LUMERICAL_HOME`, `LD_LIBRARY_PATH="/usr/local/cuda/compat:…"` (which the job scripts then strip), baked license defaults, `QT_QPA_PLATFORM=offscreen`, `QT_OPENGL=software`. Build: `apptainer build lumerical-2026R1.sif lumerical.def`.

---

## 6. IGUM differences

| | Athena | IGUM |
|---|---|---|
| Deploy | `bash athena/deploy_athena.sh` | `bash igum/deploy_igum.sh` (same flag surface) |
| Runtime | apptainer container | **native** extracted RPM tree; containers are impossible (no apptainer/singularity, no module system, docker daemon denied — verified 2026-08-12) |
| `LUM_HOME` | `/opt/lumerical/v261` (in container) | `/home/evyatarrubin/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261` |
| GPU request | `--gpus=1` | `${GPU_REQ}` = `--gres=gpu:1` |
| Account | not needed | `--account=${SLURM_ACCOUNT}` on **every** submit |
| QOS/partition | `24h_1g` + 5-partition auto-pick | `qos-preempt` + `part-preempt` (QOS name must match the partition) |
| `--gpu=<type>` | supported | **hard error** — "not supported on igum … Edit ARRAY_PARTITIONS in igum/igum.conf instead" |
| Headless display | Xvfb `:99` in container | `QT_QPA_PLATFORM=offscreen` (no Xvfb on IGUM, none needed) |
| Engine binary | `fdtd-engine-ompi-lcl` | plain `fdtd-engine` (`-ompi-lcl` fails: missing `libmpi.so.40`) |
| Paths in python | hardcoded `/work/...` | derived from `WORK_DIR` env (deploy adds `WORK_DIR=${REMOTE_BASE}` to every `--export`); `LUMAPI_PATH` env-overridable; legacy `/work/...` sidecar paths are remapped onto `WORK_DIR` |
| Sweep list path | `/work/data/sweep_list_<tag>.txt` | `${REMOTE_BASE}/data/sweep_list_<tag>.txt` |
| License DNS | hosts-file hack required; `lmstat` gives false `-96` | resolves natively; `lmstat` reliable (use IGUM as the seat-probe vantage) |
| scilibs | `$HOME/scilibs` bound to `/scilibs` | `${WORK_DIR}/scilibs` on `LD_LIBRARY_PATH` — **required**: part-preempt nodes lack libgfortran and the X/GL client libs (job 41776 died in 2 s on `import scipy.io`) |
| Preemption | every partition `PreemptMode=REQUEUE` | `part-preempt` also REQUEUE; MaxTime=UNLIMITED, MaxArraySize=1001 |
| Local results | `results_from_athena/` | `results_from_igum/` |
| `MAX_CONCURRENT` | 8 | 4 (seats shared) |

Not possible on IGUM: containers, `--gpu=<type>`, `rpm2cpio` (extract on Athena and tar-stream over the LAN), `sacctmgr` from the login node (use `sshare -U` / `scontrol show assoc_mgr`). IGUM's infrastructure is the weak link (slurmdbd down, login flaps) while its compute is fine — give it long self-contained resume-protected runs and keep interactive work on Athena. Known IGUM trap: simultaneous Lumerical starts on one node race `ansyscl` and die at 60 s → dispatch with `--max-concurrent=1`.

Both `athena/jobs|scripts` and `igum/jobs|scripts` are a **maintained pair** — any edit to one is mirrored in the same change or explicitly reported as not mirrored. `dgx/` was deleted 2026-09-11; never recreate a third fork.

---

## 7. Known failure modes and their exact signatures

| Signature (as it appears in logs) | Cause | Action |
|---|---|---|
| `ERROR: unknown flag '<x>' — aborting before any submission.` | Invented deploy flag (incl. `--watch`, `--watch-only`) | Use `--upload-only` / `--status` |
| `FATAL: lum_fdtd_solve not found in license pool.` then exit 2 | Engine job's `lmstat` pre-flight failed | Probe seats from IGUM; canary one task before the fleet |
| Log shows `Simulation time: ~1 s`, then later `Can not find result 'expansion for port monitor'` | **Athena license starvation = SILENT no-op** | Check "Simulation time" FIRST: ~1 s ⇒ license; normal solve time ⇒ shared-`.h5` clobber |
| Bare `LumApiError: 'in run:'` / "Unable to checkout" instantly | **IGUM license starvation = loud instant death** (also the 7th concurrent solve) | Keep total concurrent solves ≤6 across both clusters; resubmit dead indices with `--array-tasks=` |
| `FATAL: GPU appears unused 120s after engine start (mem=<N>MiB).` | Engine silently fell back to CPU | Fix GPU/driver, or resubmit with `REQUIRE_GPU=0` only if a CPU run is acceptable |
| `athena_run.py` exits **2** | `setresource('FDTD',1,'device type','GPU')` failed or read back non-GPU with `REQUIRE_GPU=1` | No `fdtd_gpu` seat / GPU not visible |
| `cudaGetDeviceCount Failed: unsupported display driver / cuda driver combination` | CUDA-12.2 compat shim loaded ahead of the host R570 driver | The job scripts already strip `/usr/local/cuda/compat*`; don't reintroduce it or the NVML trampoline |
| `QXcbConnection: Could not connect to display` | Missing virtual X11 | Athena: Xvfb `:99`; IGUM: `QT_QPA_PLATFORM=offscreen` |
| `free(): invalid pointer` | TBB malloc interceptor not preloaded | Keep the absolute-path `LD_PRELOAD` of `libtbbmalloc.so.2:libtbbmalloc_proxy.so.2` |
| `import scipy.io` dies in ~2 s on an IGUM compute node | node lacks libgfortran/libquadmath/X libs | `${WORK_DIR}/scilibs` must be on `LD_LIBRARY_PATH`; a clean `ldd` does NOT prove a node is OK (some libs are dlopen'ed) |
| Job hangs at `Setting --writable-tmpfs` | Home ~300 GB quota exceeded | Delete `.h5` scratch (`athena/h5_clean_once.sh`); don't keep `.h5` by default |
| `ERROR: SWEEP_INDEX=<i> out of range (file has N lines)` | A sweep list was rewritten under an in-flight/REQUEUEd task | Per-study lists prevent this; wait for queue-empty, redeploy, resubmit dead range with `--array-tasks=<lo>-<hi>` |
| `ERROR: invalid SWEEP_KIND=...` | Kind not in `VALID_KINDS` | Use the matching deploy flag |
| `ERROR: sweep list not found at <path>` | Deploy uploaded a different tag / native-vs-container path mismatch | Check `SWEEP_LIST` in the log header |
| `ERROR: fsp_list.txt not found` / `No entry at line N` | Option-1 array list missing or short | Re-upload `results/fsp_list.txt` |
| Job crashes <30 s right after a config change | **stale server code** — rsync silently skipped root `*.py` (dir perms) | Re-read the rsync itemized output; `rsync --inplace` is the known fix |
| Job stays `PD` with `(DependencyNeverSatisfied)` | The `afterok` parent FAILED | `scontrol update job <id> dependency=''` |
| `PD` with `(QOSMaxJobsPerUserLimit)` / `(Priority)` | Normal waiting | No action |
| `lmstat` returns `-96` ("lmgrd is not running") on Athena; `HOST_NOT_FOUND` locally | **False negative** — lmstat enumerates by the unresolvable FQDN while real jobs check out by IP | Do NOT block a dispatch; confirm ports 1055/2325 open by IP |
| ssh "Permission denied" with port 22 open (IGUM) | Login-node rate limit trip (~80 min of 24 polls/h did it) | Zero automated contact ≥45 min, then ONE probe; jobs are unaffected |
| Job lost hours to a REQUEUE | Every Athena partition is `PreemptMode=REQUEUE`; `contrib` QOS (prio 10000) preys on all our lanes | Any job >~2 h must persist progress incrementally and cold-start-resume (loss ≤1 eval). Resume ≥ lane choice. `MaxBatchRequeue=5` |
| `sbatch --wrap` refused | Athena `cli_filter` forbids it | Write a script file and `sbatch` it |
| Long-running process dies at ssh logout (Athena login node) | Login node kills all user processes at logout (nohup and tmux both die) | Run it as a SLURM CPU job |
| A comma list arrives truncated in a runner (`1,2,3` → `1`) | `sbatch --export` truncates comma-separated values | Pass lists via a file or repeated env vars |

QOS table (measured 2026-08-15, Athena; priority is inverse to walltime, GPU caps are per-QOS lanes that stack): `2h_2g` 2 h / 2 GPU / 3 jobs / prio 1000; `12h_4g` 12 h / 4 / 3 / 500; `24h_1g` (default) 24 h / — / 4 / 300; `24h_4g` 24 h / 4 / 3 / 250; `72h_8g` 72 h / 8 / 1 / 50; `4d_1g` 4 d / — / 8 / 50.

---

## 8. Adding a new study so it appears in the deploy menu

### 8.1 New study inside an existing category (no deploy edits needed)

1. **Copy the closest existing file** in the right directory (never scaffold):
   - sweep → `runners/sweeps/<name>.py` with a top-level `SPEC = SweepSpec(...)`; sweepable fields come from `experiment_card._CARD_FIELD_MAP` (add a field there once to make it sweepable).
   - one-shot → `runners/single/<name>.py` with `def run(cfg=None)` or `run = run_single_sim` at **column 0**.
   - TM → `runners/tm/<name>.py`, same contract.
   - optimization variant → copy `optimize_transmission.py` / `smoke_test.py` inside the family dir; base config from `make_optimization_base()`.
   - convergence → `convergence_testing/<name>.py`; add `PHASES = [...]` + `KIND_PREFIX = "..."` to get one array entry per phase.
2. **Unique label / `STUDY_DIR`** so outputs land in their own `results/<study>/results/` and `generate_file_tag()` names don't collide with a concurrent study. Make the dir name parameter-aware (e.g. pitch in the name) if the same study runs at several anchors.
3. Optional hooks: `BASE`, `PRELIM_SPEC`/`PRELIM_BASE`, `PRELIM_RUN_SCRIPT`, `LOCKED_LAMBDA_FILE` (may contain `{idx}`), `IS_PRELIM = True` (hide from the menu), `IS_HELPER = True` (hide a single/TM runner), `STUDY_DIR_NAME`, `build_cfg(cfg)`, and for lumopt2 `N_TASKS` + `main(task_idx)`.
4. Verify the menu locally before dispatch: `bash athena/deploy_athena.sh --option2` and confirm the new entry is listed; or run the list builder directly (zero GPU):
   ```bash
   python athena/scripts/build_sweep_list.py --kind spec --module runners.sweeps.<name> --output /tmp/x.txt
   ```
   The printed task count is the array size you will dispatch.
5. Menu-visibility traps:
   - Sweep menus grep for the **literal text `SPEC =` anywhere** in the file — a docstring or comment containing it also puts the file in the menu.
   - Single/TM menus need `run` at column 0.
   - A shared helper must contain **neither** a top-level `run` **nor** the literal `SPEC =` — put helpers at the `runners/` root (never scanned, e.g. `optimization_common.py`) or `_`-prefix them.
   - `rsync --delete`: deleting/moving a local file removes it from the server's `project/runners/` on the next deploy (server `results/` untouched).

### 8.2 New *category* (a new `runners/<newdir>/`) — three edits, mandatory

1. `athena/deploy_athena.sh`: add a menu line in the option-2 pipeline sub-prompt **and** a picker block (copy the `_PIPELINE_KIND == "..."` block that matches the closest family), setting `OPTION`/`SWEEP_KIND`/`SPEC_MODULE`.
2. `athena/scripts/athena_run.py` **and** `igum/scripts/athena_run.py`: add the dir to `_AUTO_DIRS` (for single-run-style categories).
3. Mirror every change into `igum/deploy_igum.sh` in the same commit (the pair rule), and — for a new array kind — add it to `VALID_KINDS` + `_DISPATCH` in both `athena_run_one.py` files and to `build_sweep_list.py`'s `--kind` choices.

**Never rename** `single/`, `tm/`, `sweeps/`, or the four optimization directories — they are hardcoded in both deploy scripts. `runners/archive/` is never scanned (archived files stay runnable by module path).

---

## 9. Quick reference card

```bash
# preflight
bash athena/deploy_athena.sh --license-probe
bash athena/deploy_athena.sh --status
bash athena/deploy_athena.sh --gpu-status
ssh igum '$HOME/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve' | grep "Users of lum_fdtd_solve"

# dispatch
bash athena/deploy_athena.sh --option2 --run=<runner>
bash athena/deploy_athena.sh --spec=runners.sweeps.<study> [--array-tasks=0] [--max-concurrent=2] [--gpu=a100] [--qos=2h_2g --time=02:00:00] [--after=<jobid>]
SBATCH_MEM=300G bash athena/deploy_athena.sh --spec=runners.sweeps.<study>
bash athena/deploy_athena.sh --upload-only
bash igum/deploy_igum.sh   --spec=runners.sweeps.<study> --max-concurrent=1

# monitor / fetch / stop
ssh evyatarrubin@athena.technion.ac.il "squeue -r -u evyatarrubin -o '%.12i %.30j %.8T %.10M %R'" | grep -vE "post-quantum|openssh|may need to be upgraded"
ssh evyatarrubin@athena.technion.ac.il tail -f /home/evyatarrubin/bragg_sim_athena/jobs/logs/lum_array-<ID>_0.out
bash athena/deploy_athena.sh --results-no-fsp
ssh evyatarrubin@athena.technion.ac.il "scancel <ID>"
sacct -j <ID> --format=JobID,TimeLimit,State,Elapsed
```

Local sanity check on any fetched `.mat`: `resonance_wavelength_nm` finite and inside the scan window; peak T above the dead-device floor (≈0.0008); `Q = resonance_wavelength_nm / |spectral_fwhm_nm|` (the stored `spectral_fwhm_nm` is often negative). If either check fails, surface it before building anything on the result.

---

## 10. Addenda: `zeus/` status and the VS Code task names

### 10.1 `zeus/` — legacy, effectively dead; never dispatch there

`zeus/` still exists in the repo (`zeus/deploy.sh`, `zeus/zeus.conf`, `zeus/jobs/{run_fsp_job.sh,run_python_job.sh}`, `zeus/scripts/server_run.py`) but it is a **third, older cluster on a different scheduler** and is not part of the live workflow:

| Fact | Value / evidence |
|---|---|
| Cluster | `zeus.technion.ac.il`, user `evyatarrubin`, `REMOTE_BASE=/home/evyatarrubin/bragg_sim` (`zeus/zeus.conf`) |
| Scheduler | **PBS (`qsub`), not SLURM**; CPU-only, `N_CPUS=80` (`select=1:ncpus=N`) |
| Last commit touching `zeus/` | `Tue May 5 2026` ("cavity sweep and downloading single files from servers") — vs `Mon Aug 31 2026` for `athena/` |
| Coverage in project rules | CLAUDE.md §1 names **only** Athena and IGUM as run targets ("Both clusters work; ASK which one"); Zeus appears nowhere in CLAUDE.md or any skill |
| Feature parity | No sweeps: "Sweeps are not supported on Zeus (no SLURM array) — use Athena for those" (`zeus/deploy.sh` comment). Parser accepts only `--option1 --option2 --upload-only --results --results-no-fsp --results-full --results-files[=…] --status --run[=] --preset= --spec=` and has **no unknown-flag guard** (bare `esac`, so typos are silently ignored) |
| Known trap | memory `project_zeus_lumerical_license.md`: a `License.ini` with `domain=1` breaks lumapi on Zeus |
| Docs | `README.md` §"Zeus Deployment (CPU / PBS)" (lines 415–455) still documents it |

Treat it as historical: do not dispatch FDTD there, do not mirror `athena/` changes into it, and do not delete it without explicit permission (CLAUDE.md §8).

### 10.2 `.vscode/tasks.json` — exact task and option labels

The file exists. Shell is `C:\Program Files\Git\bin\bash.exe -l -c`. Three tasks, each a `pickString` menu:

**Task `Athena`** (input id `athenaAction`, description "Athena — pick action:"):

| Label | Command |
|---|---|
| Deploy — FSP job (local .fsp → engine on Athena) | `bash athena/deploy_athena.sh --option1` |
| Deploy — Python pipeline (single sim or sweep) | `bash athena/deploy_athena.sh --option2` |
| TM only — run + analyze (no TE comparison) | `bash athena/deploy_athena.sh --option2 --run=run_tm` |
| TM vs TE — comparison + pitch correction | `bash athena/deploy_athena.sh --option2 --run=run_tm_vs_te` |
| Show GPU availability | `bash athena/deploy_athena.sh --gpu-status` |
| Watch job status (no deploy) | `bash athena/deploy_athena.sh --watch-only` ← **BROKEN, aborts on unknown flag; use `--status`** |
| Upload only (no submit) | `bash athena/deploy_athena.sh --upload-only` |
| Download results | `bash athena/deploy_athena.sh --results` |
| SSH into Athena | `ssh athena` |

**Task `IGUM`** (input id `igumAction`, "IGUM (ECE) — pick action:"):

| Label | Command |
|---|---|
| Deploy — FSP job (local .fsp → engine on IGUM) | `bash igum/deploy_igum.sh --option1` |
| Deploy — Python pipeline (single sim or sweep) | `bash igum/deploy_igum.sh --option2` |
| Watch job status (no deploy) | `bash igum/deploy_igum.sh --watch-only` ← **BROKEN, same reason** |
| Upload only (no submit) | `bash igum/deploy_igum.sh --upload-only` |
| Download results | `bash igum/deploy_igum.sh --results` |
| SSH into IGUM | `ssh igum` |

**Task `Zeus`** (input id `zeusAction`, "Zeus — pick action:") — legacy per §10.1:

| Label | Command |
|---|---|
| Deploy (interactive — choose mode) | `bash zeus/deploy.sh` |
| Deploy — FSP job (local .fsp → engine on Zeus) | `bash zeus/deploy.sh --option1` |
| Deploy — Python job (single sim) | `bash zeus/deploy.sh --option2` |
| Watch job status (no deploy) | `bash zeus/deploy.sh --watch-only` (silently ignored — Zeus's parser has no catch-all) |
| Upload only (no submit) | `bash zeus/deploy.sh --upload-only` |
| Download results | `bash zeus/deploy.sh --results` |
| SSH into Zeus | `ssh zeus` |

All three tasks are in group `build` with `presentation: reveal always, dedicated panel, clear true` and no problem matcher. Two of the nine Athena entries and one of six IGUM entries are stale (`--watch-only`); the menu is not a reliable source for the flag surface — §2 is.


---

# Part 3 — Code and data inventory

The knob surface, the module map, the runner tree, the tools, the result layout and the
verification gates. Its internal section numbers are local to this part.


Repo root: `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes` · branch `add-claude-rules-skills` @ `9b8de59`
Everything below was read from source this session. Items marked **UNVERIFIED** were not confirmed.

---

## 1. Core engine modules

Pipeline: `SimulationConfig` → `cfg.to_device_kwargs()` → `PiShiftBraggFDTD` (builds + solves) → `post_processing.analyze_simulation()` → `result_<tag>.mat`.

### `config.py` (29 lines) — machine paths only
Module-level constants: `USE_GPU=True`, `BASE_SAVE_DIR`, `NEFF_DATA_PATH` (TE-only FDE n_eff sweep `.mat`), `MATERIAL_DB_PATH=None`, `LUMAPI_PATH=r"C:\Program Files\Lumerical\v261\api\python\lumapi.py"`.
`__getattr__` computes `LAYOUTS_DIR` / `RESULTS_DIR` **lazily** as `BASE_SAVE_DIR/[$RUN_NAME/]{layouts,results}` and `os.makedirs` them — so setting `os.environ["RUN_NAME"]` mid-run retargets output (this is how the cluster dispatcher assigns study folders).

### `simulation_config.py` (741 lines) — THE KNOB SURFACE (complete)

`SimulationConfig` aggregates 14 sections via `field(default_factory=...)`, attribute names exactly:
`geometry, grating, apodization, spectral, mesh, material, source, symmetry, monitors, farfield, scatterer, phase_correction, run, simple_grating` + 2 loose fields.

**`GeometryConfig`** (`cfg.geometry.*`)
| field | default | unit |
|---|---|---|
| `avg_corrugation_width_m` | `800e-9` | m |
| `corrugation_depth_m` | `300e-9` | m (wide − narrow) |
| `core_height_m` | `350e-9` | m |
| `width_port_m` | `1000e-9` | m |
| `substrate_thickness_m` | `10e-6` | m |
| `n_devices` | `1` | 1 or 2 |
| `device_gap_m` | `1.0e-6` | m (edge-to-edge) |
| `device_stagger_m` | `0.0` | m |
| `corrugation_depth_2_m` | `None` | m (None → dev 1) |
| `avg_corrugation_width_2_m` | `None` | m (FW-BIC detune knob) |
| `device2_closed` | `False` | bool |

Derived properties: `width_wide_m`, `width_narrow_m`, `width_wide_2_m`, `width_narrow_2_m` (+ private `_corrugation_depth_2`, `_avg_width_2`).

**`GratingConfig`** (`cfg.grating.*`)
| field | default | unit/values |
|---|---|---|
| `pitch_m` | `500e-9` | m |
| `n_periods_each_side` | `80` | count |
| `cavity_neg_detuning_nm` | `0.0` | nm shortening from pitch/2 |
| `cavity_width_option` | `"avg"` | `"narrow"`/`"avg"`/`"avg_ext"` |
| `cavity_width_m` | `None` | m (numeric override, wins) |
| `innermost_tooth_shift_m` | `0.0` | m (legacy single-tooth) |
| `lengthen_cavity` | `True` | bool |
| `shift_target` | `"narrow"` | `"narrow"`/`"wide"` |
| `n_free_inner_teeth` | `1` | count |
| `inner_dw_nm` | `None` | list[nm], innermost first |
| `inner_shift_nm` | `None` | list[nm] |
| `enforce_mirror_symmetry` | `True` | bool (flag only) |
| `wall_phase_offset_deg` | `0.0` | deg of pitch (bottom wall) |
| `corrugation_profile` | `"rect"` | `"rect"`/`"sin"`/`"tri"` |
| `inner_tooth_shape` | `"rect"` | `rect`/`ellipse`/`tri`/`wedge_cav`/`wedge_out` |
| `n_shaped_inner_teeth` | `1` | count |
| `cavity_shape` | `"rect"` | `rect`/`barrel`/`hourglass` |
| `cavity_shape_depth_nm` | `150.0` | nm |
| `asym_inner_dw_delta_nm` | `None` | list[nm], antisymmetric |
| `width_narrow_per_tooth_m` | `None` | list[m], innermost first |
| `width_wide_per_tooth_m` | `None` | list[m] |

**`ApodizationConfig`**: `enabled=False`, `n_apod_periods_each_side=5`, `center_mod_depth_nm=100.0`, `method='linear'` (`'linear'`/`'tanh'`), `tanh_steepness=2.0`.

**`SpectralConfig`**: `center_wavelength_m=1.5601e-6`, `scan_width_nm=20.0`, `n_wl_points=3001`, `n_2d_monitor_points=51`.

**`MeshConfig`**: `n_periods_dist_to_port=20`, `n_wls_dist_port_to_pml=5.0`, `simulation_mode="optimization"`, `auto_shutoff_min=None` (builder default 1e-7). Property `cells_per_half_period` from `_MESH_MODE_CELLS = {"accurate":7, "optimization":5}`; `dx = pitch/(2*cells)`.

**`MaterialConfig`**: `use_constant_materials=True`, `n_core_const=1.97`, `n_clad_const=1.444`, `n_eff_guess=1.55`, `const_material_mode="object"` (`"object"`/`"sampled"`).

**`SourceConfig`**: `polarization="TE"` (`"TE"`/`"TM"`).
**`SymmetryConfig`**: `use_y_symmetry=True`, `use_z_symmetry=True`.

**`ScattererConfig`** (`cfg.scatterer.*`) — also the trench and in-core-hole path:
`enabled=False`, `shape="cylinder"` (`'cylinder'`/`'rect'`), `radius_m=150e-9`, `x_span_m=None`, `y_span_m=None`, `x_m=0.0`, `y_m=1.0e-6`, `index=None` (None → `n_core_const`; set `1.444` = SiO2 hole), `material=None` (named DB material, e.g. PEC/Al — overrides index), `mirrored_y=True`, `height_m=None`, `z_min_m=None` (absolute z floor → trench; requires `use_z_symmetry=False`), `x_list_m=None`, `y_list_m=None`, `r_list_m=None`, `rot_list_deg=None`.

**`MonitorConfig`**: `record_2d_fields=True`, `field_2d_x_span_m=None`, `monitor_2d_center_nm=None`, `monitor_2d_span_nm=None`, `downsample_yz=1`, `record_3d_fields=False`, `field_3d_span_m=None`, `field_profile_freq_points=None` (builder hardcodes 501).
⚠ `sim_helpers.apply_monitor_overrides` reads `cfg.monitors.n_3d_freq_points`, which is **not declared** in `MonitorConfig` — it only works because dataclasses accept unknown attributes (the CLAUDE.md §5 config-override trap). Two runners set it explicitly.

**`FarFieldConfig`**: `enabled=False`, `farfield_x_span_m=30e-6`, `farfield_dist_wls=0.8`, `ff_resolution=201`, `farfield_freq_points=1`, `save_nearfield=True`, `save_complex=False`.

**`PhaseCorrectionConfig`**: `do_length_correction=True`, `do_envelope_correction=True`.
**`RunConfig`**: `cleanup_lumerical_data=False`, `export_interconnect=False`.
**`SimpleGratingConfig`** (used by `run_simple_bragg.py` only): `n_periods_total=40`, `n_periods_dist_to_port=30`, `n_wls_dist_port_to_pml=5.0`.

**Loose top-level fields**: `span_multiplier_override=None`, `y_span_override_m=None`.
**Derived properties**: `_span_multiplier` (5.0 if farfield else 1.8), `_max_drawn_wide_m` (per-tooth arrays win over the scalar), `y_span`, `z_span` (`core_height + mult*λ_c`).
**Methods**: `to_device_kwargs()`, `to_simple_device_kwargs()`. Module function `set_nested_attr(cfg, "grating.n_periods_each_side", 120)`.

### `bragg_device.py` (1575 lines) — the builder

`class PiShiftBraggFDTD` — a ~110-kwarg `__init__` (all mapped from `to_device_kwargs()`), lumapi import with `config.LUMAPI_PATH` fallback.

Methods: `_setup_materials`, `_reset_layout`, `build()`, `_add_fdtd_region`, `_add_aligned_mesh_override`, `_add_bragg_core`, `_add_scatterers`, `_add_source_and_monitors`, `update_scan(center_lambda_m, width_nm, n_points)`, `close()`, `get_s_and_t_matrix(neff_mat_file=None, correct_length=True, correct_envelope_and_t_phase=True)` → `(wl, R, T, Loss, T_matrix, S11, S21)`.

`build()` order: reset → FDTD region → mesh override → core → scatterers → source/monitors.

Key derived attributes (read by `generate_file_tag` and post-processing): `dx_override = (pitch/2)/cells_per_half_period`, `cavity_length`, `cavity_length_effective`, `x_grating_end = N*pitch + cavity_length/2`, `dist_grating_to_port` (snapped to dx), `x_port`, `fdtd_x_span`, `y_dev1/y_dev2/x_dev2`, flags `_has_scatterer`, `_scatterer_n`, `_has_asym_dw`, `_domain_tag_active`, `_object_index_mode`.

**Solver settings hardcoded in `_add_fdtd_region`**: all-PML BCs; symmetry parity by polarization — TE → y min `Anti-Symmetric`, z min `Symmetric`; TM → y min `Symmetric`, z min `Anti-Symmetric`. `force symmetric y mesh` when y-sym on; **`force symmetric z mesh` ALWAYS** (z-mesh knife-edge fix). `dimension="3D"`, GPU if `config.USE_GPU`, `simulation time = 2000 ps` (env override `TM_SIM_TIME_PS`), `auto shutoff min = auto_shutoff_min or 1e-7`, custom non-uniform mesh with `dx=dx_override`, `dy=dz=50e-9`, no x grading, y/z grading factor 1.41421, `mesh refinement = "conformal variant 0"`, `dt stability factor 0.7`. Two-device runs assert `y min bc == "PML"`.

**Geometry features and their control paths**

| feature | control path | how it is drawn |
|---|---|---|
| corrugation depth / avg width | `cfg.geometry.corrugation_depth_m`, `.avg_corrugation_width_m` | per-tooth `L_narrow_d`/`L_wide_d`, `R_narrow_d`/`R_wide_d` rects |
| apodization envelope | `cfg.apodization.{enabled,n_apod_periods_each_side,center_mod_depth_nm,method,tanh_steepness}` | `get_mod_depth(d)`: linear `frac=(d−1)/denom`, tanh `tanh(a·2·frac)/tanh(2a)`; interpolates centre→edge depth |
| per-tooth explicit widths | `cfg.grating.width_{narrow,wide}_per_tooth_m` | take precedence over the envelope for the innermost `len()` teeth |
| freed inner DW / shift (inverse design) | `cfg.grating.{n_free_inner_teeth,inner_dw_nm,inner_shift_nm}` | override envelope for `d ≤ n_free`; `2·Σshift` absorbed into cavity |
| single innermost-tooth shift (legacy) | `cfg.grating.{innermost_tooth_shift_m,lengthen_cavity,shift_target}` | shortens narrow (default) or wide segment |
| cavity / π-shift segment | `cfg.grating.{cavity_neg_detuning_nm,cavity_width_option,cavity_width_m}` | object `cavity_<id>`; detuning → `override_cavity_length_nm` |
| cavity shape | `cfg.grating.{cavity_shape,cavity_shape_depth_nm}` | object `cavity_shaped` (barrel/hourglass half-sine) |
| inner-tooth shape | `cfg.grating.{inner_tooth_shape,n_shaped_inner_teeth}` | `<arm>_shbase_<d>` rect + `<arm>_shtooth_<d>_<wall>` polygons |
| smooth corrugation profile | `cfg.grating.corrugation_profile` | single sampled polygon `core_profile` |
| wall phase offset | `cfg.grating.wall_phase_offset_deg` | `<arm>toothT_<d>` / `<arm>toothB_<d>` at different x; forces y-sym OFF |
| antisym DW detune | `cfg.grating.asym_inner_dw_delta_nm` | left `corr+δ`, right `corr−δ` |
| scatterers / combs / pillar rows | `cfg.scatterer.*` (`x_list_m`, `r_list_m`, `y_list_m`, `rot_list_deg`) | `scatterer_<j>_<k>` circles or rects, `mesh order 1` |
| air / PEC trench | `cfg.scatterer.shape="rect"` + `x_span_m`,`y_span_m`,`z_min_m`,`height_m`,`material` | same object family, `use_z_symmetry=False` |
| in-core SiO2 hole | `cfg.scatterer.index = 1.444` (< core) | same, mesh order wins overlaps; tag gets `_hole` |
| two side-by-side devices | `cfg.geometry.{n_devices,device_gap_m,device_stagger_m,corrugation_depth_2_m,avg_corrugation_width_2_m,device2_closed}` | second guide at `y_dev2`; feed stubs `wg_left_inf`/`wg_right_inf`, `base_strip` |
| feed waveguides & ports | `cfg.mesh.{n_periods_dist_to_port,n_wls_dist_port_to_pml}`, `cfg.geometry.width_port_m` | `Port_1`(−x,fwd), `Port_2`(+x,bwd), `Port_3/Port_4` (device 2, open only); `mode selection = "fundamental <TE|TM> mode"`, `frequency dependent profile 1`, source port = Port_1 |
| monitors | `cfg.monitors.*`, `cfg.farfield.*` | `field_profile` (2D Z-normal 1D tracker, 501 freq pts), `field_profile_2D_XY`, `field_profile_2D_YZ_cross`, `field_profile_2D_XZ_side`, `field_profile_3D`, `side_monitor` (y-normal), `top_monitor` (z-normal), `mesh_override` |

`SimpleBraggFDTD` is **not** in this file — it lives in `runners/single/run_simple_bragg.py:41`.

### `sim_helpers.py` (634 lines)
| function | purpose / key args |
|---|---|
| `extract_farfield(fdtd, monitor_name, ff_res=201, idx_f=1, complex_fields=False, lam_target_m=None)` | `farfield3d/ux/uy` via `eval()`; returns `{E2, ux, uy, lam}` (+`Ex_c,Ey_c,Ez_c`); `lam_target_m` picks the recorded point nearest resonance |
| `extract_monitor_nearfield(fdtd, monitor_name)` | `{x,y,z,E_res,lambda_arr}` |
| `extract_monitor_polarimetry(fdtd, monitor_name, normal)` | server-side reduced Poynting split → `{lam,x,P_total,P_tm,P_te,prof_total,prof_tm,prof_te,flux_norm}` |
| `find_bragg_resonance(wl, T)` | **the** resonance finder: `score = (prominence/(width+1))·(1−base_level)`; falls back to `argmax` only if no peaks |
| `peak_t_diagnostic(wl, T)` | `(λ_peak, T_peak)` wrapper for inverse design |
| `calculate_fwhm_relative(x, y)` | FWHM at `y_min + 0.5(y_max−y_min)`, interpolated crossings |
| `extract_envelope_peaks(x, y)` | cubic interp through `find_peaks` |
| `extract_and_process_field_profile(sim, target_wl)` | **the one legal mode-width measurement**: `trapezoid(|E|² , y)` → crop to `±x_grating_end` → envelope → FWHM. Returns `(f_x, I_x_1D, I_x_envelope, fwhm_val, actual_wl)` |
| `generate_file_tag(sim)` | the filename tag (§5) |
| `apply_monitor_overrides(sim, cfg)` | sets frequency points on `field_profile` / 2D / 3D / far-field monitors; optional own 2D window |

### `post_processing.py` (573 lines) — 10 numbered stages
Dataclasses: `SParameters(wl,R,T,Loss,T_mat,S11,S21)`, `ResonanceResult(idx,wavelength_m,transmission,spectral_fwhm_m)`, `FieldProfileResult(x,intensity_1d,envelope_1d,fwhm_m,monitor_wavelength_m)`.
Functions: `extract_s_parameters(sim,cfg)` (FDE neff file for TE, live port neff for TM) · `find_resonance(s_params)` (`find_bragg_resonance` + `scipy.signal.peak_widths(rel_height=0.5)`) · `extract_field_profile(sim,resonance)` · `extract_2d_fields(sim)` · `extract_3d_fields(sim)` · `extract_farfield_data(sim,cfg,lam_res_m=None)` · `assemble_results(...)` · `save_results(results,results_path)` (`sio.savemat(..., do_compression=True)`) · `export_interconnect(s_params,results_dir,tag)` → `interconnect_symmetric_<tag>.txt` · `plot_results(...)` · **`analyze_simulation(sim,cfg,results_path,tag="")`** — the single entry point, call right after `fdtd.run()`.

### `analysis.py` (168 lines) — S-parameter math, no Lumerical
`align_phases_at_resonance_peak(wl,S11,S21,target_phase=0.5π)` · `apply_phase_correction(wl,S11_raw,S21_raw,pitch,dist_grating_to_port,x_grating_end, neff_mat_file=None, neff1_internal=None, neff2_internal=None, use_single_neff=False, single_neff_val=None, do_length_correction=True, do_envelope_correction=True)` — stage A feed de-embed, B `β₀=π/pitch` slope removal, C rotate to −π/2 · `calculate_physics_matrices(S11,S21)` → `(R, T, Loss=1−R−T, T_matrix)` assuming reciprocity/symmetry · `export_for_interconnect_symmetric(filename,wl,S11,S21)`.

### `experiment_card.py` (262 lines) — the sweepable-knob registry
`_CARD_FIELD_MAP`: **single source of truth** mapping short card field → `(dot path, transform)`. ~60 entries, e.g. `corrugation_depth_nm → geometry.corrugation_depth_m (×1e-9)`, `cavity_neg_detuning_nm → grating.cavity_neg_detuning_nm`, `scatterer_x_list_nm → scatterer.x_list_m`, `y_span_um → y_span_override_m`, `span_mult → span_multiplier_override`. To make a new parameter sweepable, add it **here once**.
`class ExperimentCard` (all fields `Optional`, default None = use config default) + `label`; `.to_config(base=None)`.
`run_card(card, base=None)` / `run_cards(cards, base=None)` → one `run_single_sim` per card; `_print_card_results`.
Trap in the code's own comment: `cavity_length_nm` maps to an attr `to_device_kwargs` ignores — use `cavity_neg_detuning_nm`.

---

## 2. `runners/` tree

```
runners/  README.md  __init__.py  optimization_common.py
  single/ (12)  tm/ (13)  sweeps/ (34)  scatterers/ (62)  metal_mirror/ (23)
  hole_lattice/ (2)  side_by_side/ (1)  experiment_comparison/ (8)
  visualization/ (2)  inverse_design/ (12)  fd_gradient_design/ (4)
  gradient_free_design/ (9)  lumerical_native_optimization/ (5)
  lumopt2_design/ (42 py/sh + 14 md + gates/)  archive/
```

**Deploy-menu contract** (`athena/scripts/athena_run.py`): `runners/single/*.py` and `runners/tm/*.py` must define a top-level `run = <callable>`; `runners/sweeps/*.py` must define `SPEC = SweepSpec(...)`; files prefixed `_` or setting `IS_HELPER = True` are skipped. Optional `STUDY_DIR_NAME = "..."` redirects output to `results/<name>/`; optional `build_cfg(cfg) -> cfg` hook applies overrides on both local and cluster paths. Optimizer families are found by grepping for the literal `SPEC =` in `runners/<family>/*.py`.

### ENGINES (reused — do not copy-with-tweak)
- `runners/single/run_simulation.py` — `run_single_sim(cfg, *, show_plots=True, tag_suffix="", save_figs=True)`: build → `update_scan` → `apply_monitor_overrides` → `save` → `run` → `analyze_simulation`. Writes `layout_<tag>.fsp` + `result_<tag>.mat`; appends `_3D` / `_ff` / `tag_suffix` to the tag.
- `runners/sweeps/sweep_spec.py` — `SweepSpec` + `run_sweep_spec` (below).
- `runners/single/run_experiment.py` — drives `ExperimentCard` lists.
- `runners/single/run_simple_bragg.py` — holds `class SimpleBraggFDTD` (uniform grating, no cavity).
- `runners/sweeps/_tm_base.py` — shared anchored-TM base config (historical quirk: returns scatterer **enabled** + `y_span_override_m=4.8e-6`; non-scatterer consumers must flip both).
- `runners/tm/_tm_vs_te_common.py` (`IS_HELPER`) — one single-wide-scan step shared by `run_te.py` / `run_tm.py` / `run_tm_vs_te.py`.
- `runners/scatterers/_common.py` — shared knobs/base for the Green's response-matrix program.
- `runners/optimization_common.py` — shared base config for the four optimizer families.
- `runners/visualization/plot_optimization.py` — `plot_convergence`, `plot_spectrum_overlay`.
- `runners/experiment_comparison/compare_with_experiment.py` + `it11_card_builder.py`, `device_names.csv` — sim-vs-fab comparison.

### SweepSpec — how a sweep is declared
Every field is a **list**; the swept set is exactly `_CARD_FIELD_MAP`'s keys (an unknown field raises). Behaviour fields: `mode: "cartesian"|"zipped"`, `label: str`.
```python
# runners/sweeps/<study>.py
from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
SPEC = SweepSpec(
    n_apod_periods_each_side = [0, 3, 5],
    innermost_tooth_shift_nm = [0, 50, 100, 150],
    cavity_neg_detuning_nm   = [5.76],
    label = "apod_and_shift",
)
if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local")
```
Methods: `_populated()`, `expand(base=None) -> list[SimulationConfig]` (cartesian product or lockstep zip; `apod_method=='none'` and `n_apod_periods_each_side==0` both disable apodization), `describe()` (prints fields + task count — use it as the pre-dispatch check).
`run_sweep_spec(spec, target="local"|"zeus"|"athena")` — only `"local"` is implemented; `"zeus"` and `"athena"` raise `NotImplementedError` pointing at `bash athena/deploy_athena.sh --option2` / `bash igum/deploy_igum.sh --option2`. Cluster dispatch by module: `bash athena/deploy_athena.sh --spec=runners.sweeps.<study>`.

Non-`SPEC` files in `sweeps/`: `sweep_spec.py`, `_tm_base.py`, `plot_tm_te_shift.py`, `optimize_innermost_shift.py`. All others are one-off studies (`tm_nladder_c{276,325,400}`, `te_q3db_20um`, `tm_q3db_14um_knob`, `farfield_sph_20um`, `itai_hh_*`, `tm_span_conv_c325`, …).

### One-off study dirs (live)
`single/` (12) · `tm/` (13, + `PITCH_ALIGNMENT.md`) · `sweeps/` (34) · `scatterers/` (62, + `COMB_HANDOFF.md`, `solve_response_matrix.py`) · `metal_mirror/` (23, trenches/PEC/comb-q3db, + `README.md`) · `hole_lattice/` (2) · `side_by_side/` (1) · `experiment_comparison/` (8) · `visualization/` (2).

### `runners/archive/` — counts and names only
`README.md`, `__init__.py`, and 3 subtrees: `experiment_comparison/` (2 files), `side_by_side/` (23), `sweeps/` (57). Contents are closed studies kept unedited; deploy menus never scan it.

### `runners/lumopt2_design/` — adjoint inverse design (42 code files, 14 md, `gates/`)
- **`lumopt2_design.py` (3302)** — the engine. 191-param vector (25 corr/avg/shift + 57 comb r/x + `d_comb` + cavity), windowed softmax-p=12 T FOM + ρ deadband, projected null-space step. Public drivers: `run_campaign`, `run_projected`, `run_canary`, `run_validate_gradient`, `run_adjoint_only`. No `__main__` — always dispatched by spec module.
- **`validate_c325.py` (1401)** — gates B0–B4, `N_TASKS = 54` (task index = experiment). Locally: `python -m runners.lumopt2_design.validate_c325` runs B0+B1 only and exits 0/1. **Tasks 47 (projected lanes) / 50 (ns2 lanes) are the mandatory ~1.5–2 h pipeline smoke** before any engine change.
- **`best_designs.py` (371)** — the named measured 191-param vectors (`BEST_T9636`, `BEST_D1_T9676`, …). Import from here; never re-paste.
- **`gpu_probe.py` (237)** — FieldRegion-on-GPU size limit on an empty box (the cheap-scene lesson).
- Campaigns: `campaign_c325_seedA{,2,3,4}.py`, `campaign_c325_seedB{,2}.py`, `campaign_c325_bare.py`, `campaign_v2_proj.py`, `campaign_v2_proj_best.py` (b1), `campaign_v2_proj_d1.py` (d1), `campaign_v2_proj_d1u.py` (d1u), `campaign_v2_projection.py`, `campaign_v2_uniform.py`, `campaign_v2_noshift.py`, `campaign_v2_seesaw.py`.
- Ladders/probes: `shift_ladder`, `elong_ladder`, `seesaw_ladder`, `corr_profile_ladder`, `prod_q3db_ladder`, `comb_basin_scan`, `comb_count_scan`, `comb_dip_ab`, `rho_neutral_shape`, `sigma_neutral_probe`, `tangent_probe_c325`, `retrim_best_c325`, `retrim_decompose`, `fit_c_field`, `fwhm_audit`, `fsp_width`, `seed_width_audit`, `prod_confirm`, `extract_spectrum`, `export_gds`.
- `dispatch_campaign.sh` — `bash runners/lumopt2_design/dispatch_campaign.sh seedA|seedB` (header says PARKED as of 2026-08-15).
- Dispatch form: `SBATCH_MEM=300G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.<module>`. (`ARRAY_TIME` as env is silently ignored — `LUMOPT2_TIME` is the working knob.)
- **State/log naming**: `<out_dir>/<label>_evals.jsonl` (per-eval), `<label>_optstate.json` (resume state), `<label>_proj.jsonl` (projected steps), `<label>_best.json`. Warm start reads `_evals.jsonl` via `_best_from_log`.

### Other optimizer families
- `runners/inverse_design/` (lumopt v1): `inverse_design.py` (1453, `InverseDesignSpec` + driver), `optimize_transmission.py`, `optimize_transmission_outer.py`, `smoke_test.py`, `scan_cavity_width.py`, `check_gradient_test{,_regular,_small_dx,_fine_mesh}.py`, `test_geometry.py`, `plot_run.py`, `STATUS.md` (verdict: gradient effectively zero — path not productive). Invocation: `python -m runners.inverse_design.inverse_design --spec runners.inverse_design.optimize_transmission`.
- `runners/fd_gradient_design/`: `fd_gradient_design.py` (646, scipy L-BFGS-B + central differences reusing the PSO evaluator), `optimize_transmission.py`, `smoke_test.py`.
- `runners/gradient_free_design/`: `gradient_free_design.py` (782, drives `addsweep("Optimization")` PSO/GA from lumapi; `freed_group` parametric structure group), `optimize_transmission{,_corrected,_tm,_tm_shift}.py`, `smoke_test.py`, `local_smoke.py`, `test_geometry.py`.
- `runners/lumerical_native_optimization/`: `lumerical_native_optimization.py` (591, built-in PSO sweep), `optimize_transmission.py`, `smoke_test.py`, `plot_run.py`.

---

## 3. `python_tools/` (20 tools + `archive/` 8; 4949 py lines)

No `__init__.py` — tools `sys.path.insert` the repo root. Most are **knob-at-the-top, no-CLI** scripts (CLAUDE.md §11).

| file | purpose | CLI | output |
|---|---|---|---|
| `analyze_batch.py` | batch-read every `result_*.mat` in a study: resonance λ (finder, not argmax), peak T, loss, `Q=λ/|spectral_fwhm_nm|`, `fwhm_m`, in-window + dead-floor flags (`DEAD_FLOOR=0.02`) | `python python_tools/analyze_batch.py <results_dir> [--tsv out.tsv]` | stdout table (+TSV) |
| `predict_q3db.py` | **q3db predictive engine** (see below) | none (edit knobs) | stdout prediction |
| `calibrate_q3db.py` | fit per-family params from stored results + backtests B1–B14 | `python python_tools/calibrate_q3db.py` | overwrites `python_tools/q3db_calibration.csv` + stdout report |
| `bragg_cmt.py` | CMT transfer-matrix engine for π-shift gratings | `python python_tools/bragg_cmt.py` → `_selftest()` | stdout; library for `calibrate_q3db` |
| `comb_kspace_model.py` | k-space interference model of the cladding post comb | `python python_tools/comb_kspace_model.py` (from repo root) | stdout only |
| `farfield_multipole.py` | vector-spherical-harmonic decomposition of a stored far field | `python python_tools/farfield_multipole.py result_..._ff.mat [...] [--lmax 200] [--csv]` | stdout + `<mat>_multipoles.csv` |
| `antineedle_comb_design.py` | zero-GPU anti-needle comb design (calibrated on job 125285) | `python python_tools/antineedle_comb_design.py` | stdout + `docs/antineedle_comb_design.mat` |
| `lateral_radiation_theory.py` | zero-license slab/EIM/TMM theory, sections A–D | `python python_tools/lateral_radiation_theory.py` | stdout tables + verdict |
| `theory_innermost_recycling.py` | leaky-field + innermost-tooth cancellation ceiling | `python python_tools/theory_innermost_recycling.py` | `docs/theory_innermost_recycling_2026-07-08.{png,pdf}` |
| `derive_boundary_profile.py` | shifting-boundary δ(x) solve, mode A (axis field + EIM) | run directly (no `__main__` guard) | stdout δ table |
| `derive_boundary_profile_stack.py` | same, mode B (real 2D field of the W1050 stack) | `python python_tools/derive_boundary_profile_stack.py` | stdout δ table |
| `farfield_export.py` | Lumerical-style far-field figures from an `.fsp` (`hide=True`) | none (constants at top) | figures + `<fsp>_farfield_export.mat` |
| `analyze_farfield.py` | live-session far-field extraction + 3D analysis | none; paths hardcoded | stdout + figures (**not** hidden) |
| `calc_neff_vs_wl.py` | `NeffSweeper` — FDE n_eff(λ) sweeps | none | `FDE_sweep_results.mat`, `.lms`, figure |
| `calibrate_n_core.py` | solve `n_core` matching an experimental Bragg λ | `python python_tools/calibrate_n_core.py` | stdout + `...\neff_calibration\calibrate_n_core_log.json` (outside repo) |
| `recommend_cavity_length.py` | FDE-based recommended cavity negative detuning | `python python_tools/recommend_cavity_length.py` | stdout |
| `overlap_analysis.py` | field-overlap integral long vs short device | none; absolute paths into another repo | stdout + figure |
| `plot_loss_spectra.py` | T/R/loss plots from `.npz` | none — **tkinter dialogs** (opens windows; conflicts with the silent-runs rule) | `combined_loss.png`, `<name>_TRloss.png` |
| `q3db_calibration.csv` | the calibration table (109 lines) | data | header `family,param,value,n_rows,source,engine_version,numerics`; families `errband, itai_te, itai_tm, knob_te, knob_tm, te_q3db_c250, tm_bare_c276, tm_bare_c325, tm_bare_c448, tm_invdesign, tm_trench_c325`; stamped `2026R1.3-b4572`, `y8.0/z8.8 box, 20nm/4001pts, dx50 conformal, ASL 1e-7` |
| `Q3DB_PREDICTOR_HANDOFF.md` | 233-line self-contained handoff for the q3db engine | doc | — |
| `archive/` | 8 closed phase-0 theory gates (`phase0_aniso_cladding`, `phase0_bic_cmt`, `phase0_cladding_reflector`, `phase0_counterdiabatic`, `phase0_greens_cluster`, `phase0_greens_overlap`, `phase0_kerker_mie`, `phase0_supercavity_fw`) + `README.md`; run as `python python_tools/archive/phase0_<name>.py`, verdicts in `docs/phase0_gate_verdicts_2026-07-06.md` | | stdout (figure output UNVERIFIED) |

**`predict_q3db.py` detail** — model: exact two-port algebra + per-family exponential `Qc(N)` + saturating power-law `Qi(N)` + width truncation fit. Families are single-polarization by construction (`tm_*`/`te_*`/`itai_*`); never mix TM and TE anchors. `MODE` ∈ `observe` (observables of `FAMILY` at `N`), `design` (find N* for `TARGET_DB`, optional `TARGET_WIDTH_UM` via the corrugation knob), `extend` (anchor levels on ONE new measured `ROW`, borrow shape from `BASE_FAMILY`), `compare` (`MEASURED` vs predicted, INSIDE/OUTSIDE the hold-out band). Knobs `FAMILY, N, TARGET_DB, TARGET_WIDTH_UM, ROW, BASE_FAMILY, MEASURED, COMPARE_FROM_ROW`; constants `CORR_QI_EXP_TM=-2.9`, `FINF_CORR_EXP=-1.11`, `QC_H_PER_NM=-0.002818`, `KAPPA_C325_TM=0.0353e6`, `PITCH_TM_M=516.83e-9`, `PITCH_TE_M=500e-9`, `Q_ADEQ=5e4`. Reads `python_tools/q3db_calibration.csv`; writes nothing; prints two error bars (model sensitivity + expected deviation) and the confirmation-run spec. The `MODE` comment lists only three modes while `main()` also implements `compare`.
⚠ `calibrate_q3db.py`'s docstring advertises `--csv out`, but there is no argparse/`sys.argv` in the file — the CSV is written unconditionally. Stale docstring.

---

## 4. `matlab_plotting/` and `matlab_analysis/`

`startup.m` (repo root) adds `.`, `matlab_plotting`, `matlab_plotting/studies`, `matlab_analysis` to the path (not `legacy/`); no `savepath`.

### Engines — `matlab_plotting/` (6, all `uigetfile` + `plot_prefs.mat` driven)
| file | lines | role |
|---|---|---|
| `plot_transmission.m` | 582 | T / R / loss / phase from `.mat`; the resonance + Q reference implementation |
| `plot_farfield.m` | 776 | per-monitor near-field ⟷ far-field pair; TE-over-TM compare mode |
| `plot_field_3d.m` | 1129 | 3D volume: XZ, XY, 3×3 YZ panel, isosurface + Poynting quiver |
| `plot_field_poynting_zoom.m` | 895 | zoomed \|E\|² (dB) + Poynting arrows (superset of the two `legacy/` scripts) |
| `plot_mode_profile.m` | 470 | mode envelope + spatial FWHM side-by-side |
| `plot_resonance_vs_param.m` | 250 | parses the swept parameter from filenames; λ_res / peak T vs parameter |

Engine-adjacent (general, non-picker): `plot_mode_profile_xz.m` (282, width from the XZ monitor), `plot_transmission_compare.m` (69), `plot_convergence.m` (263), `save_figures_interactive.m` (143). Superseded: `legacy/plot_field_poynting.m` (547), `legacy/plot_field_poynting_overlay.m` (415).

### Per-study one-offs
31 still live at the top of `matlab_plotting/` (each header names study dir + job IDs + date — e.g. `plot_comb_q3db.m`, `plot_trench_flush_q3db.m`, `plot_q20um_3db_benchmark.m`, `plot_itai_hh_*.m`, `plot_scat_*.m`, `plot_trench_*.m`); **71 archived in `matlab_plotting/studies/`** (+ `README.md`: "each script's header states the study directory and job ID it plots; when a new study closes, its plot script moves here"). Also tracked figure artifacts against §7: `matlab_plotting/results_from_athena/scat_rect_comb/comb_phase_convention.{fig,png}` and `studies/{plot_compare_8_devices.fig, plot_te300_vs_tm400_match.fig, .png}`.

### `matlab_analysis/` (7 files, 2028 lines)
`analyze_core_k_space.m` (311, FFT of the core profile, radiation vs bound region) · `analyze_farfield_radiation.m` (168) · `analyze_radiation_recycling.m` (560, outward flux vs distance) · `analyze_yz_circle_power.m` (148) · `compare_simulations.m` (328). `overlap_analysis_bg.m` (258) and `overlap_analysis_many.m` (255) are imports from another project with absolute paths outside this repo — they will not run as-is; `analyze_farfield_radiation.m` also hardcodes such a path.

### Headless invocation
```powershell
& "C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "cd('c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\matlab_plotting'); <plot_script>"
```
(`.claude/skills/fetch-results/SKILL.md:36`.) Per-study scripts also use `matlab -batch "run('matlab_plotting/<script>.m')"`. Save idiom: `figure('Visible','off')` → `savefig(fig, ...fig)` → `exportgraphics(fig, ...png, 'Resolution', 150)`.
⚠ The 6 engines call `questdlg`/`uigetfile` with **no** `batchStartupOptionUsed` guard anywhere in `matlab_plotting/*.m`, so whether the `-batch` recipe completes against an engine is **UNVERIFIED**. The `studies/` one-offs are the genuinely headless ones.

### Conventions encoded
- **Resonance**: `findpeaks(..., 'WidthReference','halfprom')` then `score = (p./(w+1)).*(1 − (pks−p))`, `argmax` fallback — the MATLAB port of `find_bragg_resonance` (`plot_transmission.m:359-375`; needs the Signal Processing Toolbox).
- **Spectral FWHM / Q**: walk **outward** from the peak to the first local half-max crossing, interpolate; `Q = lambda_res/FWHM`. The in-file bug comment records the window-edge bug that inflated TE FWHM 0.96→7.5 nm and collapsed Q 1640→208.
- **Spatial FWHM**: Hilbert envelope → `smoothdata('gaussian',50)` → relative half-max between min and max (`plot_mode_profile.m:392-421`).
- One-offs read the stored fields and apply `abs(d.spectral_fwhm_nm)` (stored negative) and key monitors off `data.resonance_wavelength_nm`.
- **Titles**: engines use plain fixed strings; the "dimensions + λ_res + peak T" convention lives in the one-offs, with real TeX `\pi`, `\mum`, `\lambda`. Q goes in the **legend** (`'%s (Q ~ %.2f x 10^%d)'`), mode width in an on-axes annotation.
- `'Interpreter','none'` appears in exactly one file, `plot_transmission.m` (5 sites, all filename-ish text); elsewhere `'tex'`.
- **View naming is split in the code.** CLAUDE.md §8 mandates XZ = "Top view", XY = "Side view"; only `plot_scat_i_fieldmaps.m` implements that. Every engine, `legacy/`, all of `matlab_analysis/`, and `plot_trench_n150_maps.m` (with a "standard convention, user-set 2026-07-21" note) use the **standard** XY = Top / XZ = Side. Axis orientation is consistent regardless: x horizontal, y or z vertical.

---

## 5. Result data layout

Standard shape per study: `<root>/<study>/layouts/layout_<tag>.fsp` (+ `.log`) and `<root>/<study>/results/result_<tag>.mat`. The study folder name = `RUN_NAME` = the runner's module name, or its `STUDY_DIR_NAME` override, or `<module_short>/<card.label>` for card sweeps.

- **`results_from_athena/`** — 209 subdirs. Tree histogram: 2804 `.mat`, 2286 `.log`, 299 `.png`, 157 `.fig`, 144 `.jsonl`, 143 `.npz`, 66 `.py`, 55 `.fsp`, 52 `.json`, 23 `.md`. Loose top-level files (10): `LOSS_EXPLORATION_FINDINGS.md`, `_cards_manifest.json`, `_prelim_sweep_list.txt`, `_sweep_list.txt`, `coupling_vs_stagger_lobe_test.png`, `lumopt2_logs_seedA4_evals.jsonl`, `mem_report_95855.txt`, `scat_c4_w1050_lambda_res.json`, `scat_greens_lambda_res.json`, `scat_h_retrocomb_lambda_res.json`.
  Subdir names (all 209): `_SAFE_BACKUP_20260628, air_trench_dscan, air_trench_w1050, anti_moment_cavity, asym_dw_study, auto_shutoff_convergence, barrel_followup, campaign_c325_seedA, campaign_c325_seedA2, campaign_c325_seedA3, campaign_c325_seedA4, cavity_acc_confirm, cavity_combo_study, cavity_design_study, cavity_hann_sweep, cavity_shape2_study, cavity_width, cavity_width_ladder, check_gradient_test, check_gradient_test_fine_mesh, check_gradient_test_regular, check_gradient_test_small_dx, comb_basin_scan, comb_count_scan, comb_dip_ab, comb_physics_rethink, comb_q3db, comb_q3db_layouts, comb_q3db_lock, compare_8_devices, compare_tm_optimized_shift, d1_generation, device_gallery, devices, distributed_shift_study, engine_canary, farfield_sph_20um, fd_gradient_design, fsp_exports, fsp_width, fwhm_audit, gradient_free_design, inline_two_defect_tm, inner_shape_study, invdesign_q3db_20um, inverse_design, it11_devices_500, it11_devices_516, layouts, linear_apod_10p_no_shift, loss_program_presentation, lumerical_native_optimization, lumopt2_c325_logs, lumopt2_v2_proj_c1, mesh_convergence, mesh_convergence_te_cmp, mesh_convergence_tm, metal_mirror_dscan, new_effects_pres, optimize_transmission, optimize_transmission_corrected, optimize_transmission_outer, optimize_transmission_tm, optimize_transmission_tm_shift, pec_trench_geom, q20um_3db_benchmark, r12_canary, radiation_kspace_diag, retro_bank, run_convergence, run_experiment, run_simulation, run_te, run_tm, run_tm_vs_te, scan_cavity_width, scat_a_baseline, scat_b_gates, scat_c2_row2, scat_c3_ygrid, scat_c4_w1050row, scat_c_response, scat_e_round2, scat_e_validate, scat_f_lattice, scat_g_apod, scat_h_retrocomb, scat_i_fieldmaps, scat_longcomb, scat_longpair, scat_n_heights, scat_o_comb1800, scat_offcentre, scat_offcentre2, scat_p_antineedle, scat_pair_apod, scat_pair_transfer, scat_r_aim536, scat_rect_comb, scat_rect_comb_tmp, scat_s_refine, scat_t_confirm, scat_u_flushcomb, scat_v_apodcomb, scat_w_dscan, scat_x10_incore_r50_axis, scat_x11_incore_axis_r30_40_60, scat_x17_incore_axis_eqwidth, scat_x18_axis_r60_period_c536, scat_x20_incore_axis_eqwidth_r30_40, scat_x21_axis_l527_athena, scat_x23_axis_l527_r50phase_r30eq, scat_x2_incore_circle, scat_x3_incore_lamscan, scat_x4_incore_below530, scat_x5_incore_r110, scat_x6_incore_r50, scat_x7_incore_r40_r30, scat_x8_incore_r50_c477, scat_x9_incore_r50_phase, scat_y_polish, scatterers_and_2pishift_presentation, shape_study, shift_ladder, si_substrate_check, side_by_side_coupling, side_by_side_debug, side_by_side_detune, side_by_side_recycle, side_by_side_te_300nm, side_by_side_tm_400nm, side_by_side_tm_400nm_widegap, side_by_side_tm_detune_400nm, side_by_side_tm_detune_400nm_ext, side_by_side_tm_detune_400nm_lobe, side_by_side_tm_detune_400nm_stag6, sigma_neutral_probe, single_sim, smoke_test, tangent_probe_c325, tanh_apod_10p_no_shift, te_q3db_20um, te_span_z_check, te_transfer_check, tm_air_trench, tm_air_trench_regular, tm_air_trench_w400fill, tm_apod_518_a20, tm_apod_pitch518, tm_bic_kerker_batch1, tm_cd_profile_scan, tm_center_completion, tm_cladding_reflector, tm_comb_box_c325, tm_derived_profile, tm_exotic_recycle, tm_field_export, tm_fw_bic_scan, tm_h200_w1800_p504_locator, tm_h200_w1800_p504_n1300, tm_hole_lattice, tm_hole_scan, tm_match_bisect, tm_match_corr, tm_novel_phase2, tm_pareto_stack_vs_apod, tm_periods_match_te, tm_pitch_match_H200, tm_radiation_polarimetry, tm_scatterer_acc, tm_scatterer_array, tm_scatterer_demo, tm_scatterer_r80_xscan, tm_scatterer_radius, tm_scatterer_scan, tm_scatterer_ydist_scan, tm_shift_c400, tm_shift_frontier, tm_shift_p518, tm_span_conv_c325, tm_span_convergence, tm_span_convergence2, tm_strip_reflector, tm_superlattice_2L, tm_te, tm_te_apod, tm_te_apod_tanh, tm_te_pitch_matched, tm_te_shift, tm_wide_mode, tm_wide_mode_H200, tm_wide_mode_H200_P546, tm_width_lightline, trench_d_refine, trench_flare_apod, trench_flush_q3db, trench_flush_top, trench_h350, trench_h4, trench_n150_full, trench_n150_h350, trench_te_apod, v2_gpu_gradient_pause, v2_lam_chain_toy, v2_ns2_toy, validate_c325, validate_te_scaling, verify_pso_best, width_envelope_study`.
- **`results_from_igum/`** — 32 subdirs: `auto_shutoff_convergence, autoshutoff_qspan, campaign_c325_seedB, elong_ladder, engine_canary, invdesign_q3db_20um, layouts, lumopt2_logs, lumopt2_v2_proj_b1, mesh_convergence, run_simulation, scat_aim_extend, scat_air_comb, scat_q_r80phase, scat_te_comb, scat_x12_incore_axis_r80_110, scat_x19_axis_r60_phase524, scat_x22_axis_l527_igum, scat_x24_axis_l527_r50eq, scat_x_incore, scat_z_teffmap, si_substrate_check, single_sim, tm_nladder_c276, tm_nladder_c325, tm_nladder_c400, tm_q3db_14um_knob, trench_apod20, trench_flare_apod, trench_flush_q3db_ctrl, trench_n150_hscan, trench_q3db_20um`. 20 loose files (`_sweep_list.txt`, `_night.txt`, `campaign_c325_bare_evals.jsonl`, `campaign_c325_seedB2_evals.jsonl`, `hh_asdrawn_spatial_T{E,M}.mat`, `itai_hh_*` figs/CSVs).
- **`results/`** — 3 subdirs, no loose files: `linear_apod_10p_no_shift`, `optimize_transmission_corrected` (has an extra `results/accurate/` level), `tanh_apod_10p_no_shift`.
- 23 `FINDINGS.md`-style docs live inside `results_from_athena/` (one per closed study); no README in any result tree. Naming reference is `FILE_NAMING.md` at the repo root.

### Filename convention — `sim_helpers.generate_file_tag(sim)` (lines 323–582)
```
apod:     N{N}_A{Napod}[_th][_M{mod}]{cav}{shift}{fc}{pol}{mat}{wgd}{2dev}{wc}{prof}{ish}{dsh}{ptw}{wp}{nosym}{dom}{scat}
no apod:  N{N}{cav}{shift}{fc}{pol}{mat}{wgd}{2dev}{wc}{prof}{ish}{dsh}{ptw}{wp}{nosym}{dom}{scat}
```
Every token is `""` unless its condition fires — the invariant is that legacy filenames never change. Tokens: `_A20` · `_th` · `_M125` (mod ≠ 100 nm) · `_D5p76` (|pitch/2 − cavity_length| > 0.01 nm, `.`→`p`) · `_S90`/`_S90w` (`w` = `shift_target=='wide'`) · `_fc` · `_TM` · `_d` (dispersive) · `_W1050` / `_avgx` / `_avg` / `""` · `_2pishift_p518p3_Ygap{}nm_Xstag{}nm_corr1{}nm_corr2{}nm[_avg2W{}][_closed]` · `_Wavg1000_C500` (avg ≠ 800 nm) or `_C250` (corr ≠ 400 TM / 300 TE) · `_profsin`/`_proftri` · `_ish{shape}{n}` `_cav{shap}{nm}` `_adw{v}d{tooth}` · `_dsh2S40s20` · `_ptw{n}W{}to{}` `_ptn{n}W{}` · `_wp45` · `_nosym` · `_Ybox6p8_Zbox8p8` `_AS1em8` · scatterer: head `_scRECT_L{len}xW{wid}` | `_scR{rmin}to{rmax}` | `_scR{r}`, then `_arr{n}_X{x0}to{x1}_Y{...}_C{corr}` or `_X{x}_Y{y}`, plus `_pair`, `_hole` (index < core), `_PEC`/`_Al`, `_H{nm}`, `_Zmin[m]{nm}`. Negative nm values use an `m` prefix.
`run_single_sim` then appends `_3D` (3D monitor), `_ff` (far-field), and any `tag_suffix`. Files: `layout_<tag>.fsp`, `result_<tag>.mat`, `interconnect_symmetric_<tag>.txt`.
Real examples: `result_N80_TM_avg_Ybox6p8_Zbox8p8.mat` · `result_N80_TM_W1050_ptw2W1030to970_Ybox6p8_Zbox8p8.mat` · `result_N80_TM_avg_Ybox16p0_Zbox8p8_scRECT_L84000xW800_X0_Y1800_pair_hole_ff.mat` · `result_N80_TM_avg_adw60d3_Ybox6p8_Zbox8p8.mat`.

### Fields inside a `result_*.mat` (`post_processing.assemble_results`, saved by `save_results`)
Always present: `wl_m, wl_nm, T, R, loss, T_matrix, S11_complex, S21_complex, resonance_wavelength_nm, resonance_transmission, spectral_fwhm_nm, L_device, pitch_m, n_periods_each_side, core_height_m, avg_corrugation_width_m, corrugation_depth_m, field_x, field_energy_density_1D, field_envelope_1D, fwhm_m` — plus, always written with inactive defaults: `scatterer_r_m, scatterer_x_m, scatterer_y_m, scatterer_mirrored_y, scatterer_n, scatterer_n_sites, wall_phase_offset_deg, corrugation_profile, inner_tooth_shape, n_shaped_inner_teeth, cavity_shape, cavity_shape_depth_m, asym_dw_delta_nm`.
Conditional: `scatterer_x_list_m`, `scatterer_y_list_m` (site list set) · `coupling_left, coupling_right, coupling_total, loss_4port, n_devices, device_gap_m, device_stagger_m, corrugation_depth_2_m, device_separation_m` and `S31_complex, S41_complex` (two-device) · `field_xy, field_yz_cross, field_xz_side` (structs `x,y,z,E_res,lambda_3d` [+`P_res`]) · `field_3d` (same keys) · `farfield_config` (`{ff_res}`), `farfield_side`, `farfield_top` (`E2,ux,uy,lam` [+`Ex_c,Ey_c,Ez_c`]), `polarimetry_side`, `polarimetry_top` (`lam,x,P_total,P_tm,P_te,prof_total,prof_tm,prof_te,flux_norm`), `nearfield_side`, `nearfield_top` (`x,y,z,E_res,lambda_arr`). Empty far-field extractions are skipped, so the key is absent.
`run_simple_bragg.py` writes a smaller set (`wl_m, wl_nm, T, R, loss, T_matrix, S11_complex, S21_complex, L_device`). `_EZSLICE.mat` files are server-side-reduced field slices (which script writes them is **UNVERIFIED**).

---

## 6. Verification / gate infrastructure

### `debug_fsp_compare/` — scene regression (zero GPU, build-only design license)
```
python debug_fsp_compare/scene_snapshot.py --out debug_fsp_compare/snapshots   # regenerate references
python debug_fsp_compare/scene_snapshot.py --out <tmp>
diff -r debug_fsp_compare/snapshots <tmp>                                      # PASS = empty diff
```
Six reference configs spanning the builder's code paths, each a committed text inventory: `snapshots/te_baseline.txt`, `te_apod_tanh.txt`, `te_shift_innersize.txt`, `tm_anchored_scatterer.txt`, `tm_shaped_cavity.txt`, `two_device.txt` (builders `_te_baseline`, `_te_apod_tanh`, `_te_shift_innersize`, `_tm_anchored_scatterer`, `_tm_shaped_cavity`, `_two_device`). Floats rounded to 1e-15 m so dumps are stable. Needs `PYTHONIOENCODING=utf-8` on Windows. Regenerate references only when a geometry change is intended, and say so.
Also here: `diff_fsp.py` (baseline vs lumopt-forward `.fsp` setting diff, run inside the Athena container), `test_single_lambda.py`, `test_lumopt_broadband.py` (the historical single-λ-source vs polygon discriminators), `baseline.fsp`, `lumopt_forward.fsp`, `diff_fsp_job.sh`.

### `runners/lumopt2_design/gates/` — run ALL FOUR before any lumopt2 dispatch
```
python runners/lumopt2_design/gates/run_all_gates.py
```
PASS = `ALL FOUR GATES GREEN — safe to dispatch`, exit 0. FAIL = `*** GATE FAILED: <g> — do NOT dispatch ***`, exit 1. Individually:
| gate | command | PASS |
|---|---|---|
| `gate_projection_local.py` | `python gate_projection_local.py` | per-check `PASS `/`FAIL `, ends `ALL PASS`, exit 0 (real `_proj_step`/`make_fct_v2`/`CampaignSpec` on synthetic gradients, no lumapi) |
| `gate_lam_chain.py` | `python gate_lam_chain.py` | all `OK  ` lines (matched estimator <1 % for every k, sign of the red shift, `gLam` invariant to λ-grid reversal at 1e-9) and the old unswapped recipe still refuses a reversed grid; exit 0 |
| `gate_lam_chain_plumbing.py` | `python gate_lam_chain_plumbing.py` | all `OK  `: `x[0]` is a scalar, one-hot jacobian, d1 spec fct on the flat `x`, **and** the anti-no-op assertion (`FAIL the old x[0][i] form did NOT raise -- gate has no teeth`); exit 0 |
| `predispatch_check.py` | `python predispatch_check.py` | every seed/detune vector inside `param_bounds(spec)` + the programmatic task-index reachability check; exit 0 |
Non-gating helpers: `derive_dwdlam.py` (`python derive_dwdlam.py` — re-derives `wg_dwdlam = 0.3655 µm/nm`, prints `OK` within 2 % else `DRIFT`; no exit code) and `h5_gate.py` (`python3 h5_gate.py gfr_full gfr_yhalf gfr_quart` on the Athena login node — `alive max|E|=…` = PASS, `DEAD (all-zero)` = FAIL; no exit code).
Above the local gates: the **pipeline smoke** is `validate_c325` **task 47** (projected lanes) / **task 50** (ns2 lanes) — same 191-param spec on an N=60 low-Q surrogate, ~1.5–2 h; its numbers are never quoted as physics. `python -m runners.lumopt2_design.validate_c325` runs the local B0+B1 half and exits 0/1.

### Other pre-dispatch gates
- Gradient correctness on new methods: `runners/inverse_design/check_gradient_test{,_regular,_small_dx,_fine_mesh}.py` — hard gate on small `vec_error`.
- Geometry round-trips without Lumerical: `runners/inverse_design/test_geometry.py`, `runners/gradient_free_design/test_geometry.py` (`python -m runners.gradient_free_design.test_geometry`).
- Optimizer smokes: `smoke_test.py` in each of the four families; `runners/gradient_free_design/local_smoke.py` (Windows, ~5–8 min).
- Theory selftest: `python python_tools/bragg_cmt.py` → `bragg_cmt selftest: all gates pass` (G5 Qc ratio, G6 envelope FWHM, G7 reciprocity, asserts < 1e-9).
- Config-override verification: after any direct attribute set, print `SPEC.expand()` / `describe()` or the build values (dataclasses silently accept unknown attributes).

### `convergence_testing/` (3 files)
- `run_mesh_convergence.py` — coordinate-descent over Phase X `cells_per_half_period ∈ [4,5,6,7,8,10]` then Phase YZ `dyz_max_step_nm ∈ [60,50,40,30,20]`; metric `"Q"` (2 % threshold) or `"lambda"` (1 %); early-stops after two consecutive within-threshold values. Checkpoint `checkpoint_p{N_PERIODS}_{METRIC}.json`. Local: `python convergence_testing/run_mesh_convergence.py`; aggregate an Athena array: `python convergence_testing/run_mesh_convergence.py --aggregate X` / `--aggregate YZ` (normally auto-submitted `--dependency=afterok`). Module constants `PHASES` / `KIND_PREFIX` are read by the deploy script's Convergence picker (`SWEEP_KIND=mesh_conv_x` / `mesh_conv_yz`).
- `run_auto_shutoff_convergence.py` — sweeps `auto shutoff min`; **hard requirement: every task must terminate via auto-shutoff (`status==2`)**, time-limited tasks are marked invalid and excluded. `python convergence_testing/run_auto_shutoff_convergence.py`, aggregate with `--aggregate SHUTOFF`.
- `run_convergence.py` — far-field monitor-distance convergence: multiple top/side monitors in one sim, all far/near field into one `.mat`.
Convergence `.mat` results are keep-forever data (CLAUDE.md §7 exception).

---

## 7. Docs in the repo

**Root**: `CLAUDE.md` (the always-on project rules) · `README.md` (architecture, quick start, config reference, apodization/shift/card/sweep/post-processing/MATLAB/convergence/Zeus sections, output format) · `FILE_NAMING.md` (the tag abbreviation table + where tags are generated).

**`docs/`** (101 files; 12 `.md`, rest figures/decks/PDFs):
| file | purpose |
|---|---|
| `RESUME_batch1_2026-07-07.md` | exact-command resume after the VPN dropped mid-run (BIC/Kerker/CD program) |
| `calibration_dossier_2026-09-26.md` | TE overshoot seed + plain TE device + TM calibration points, every number labelled MEASURED/DERIVED/EXPECTED |
| `comb_physics_rethink_2026-09-11.md` (+ `.html`, `.pdf`) | the cladding comb rethought from the physics: what recycles radiation and what cannot |
| `loss_program_bic_scatterer_2026-07-06.md` | standalone brief for the TM loss BIC/scatterer/cross-field program |
| `loss_reduction_research_2026-07-03.md` (+ `.pdf`) | literature survey on reducing TM radiation loss |
| `novelty_analysis_2026-07-07.md` | what is genuinely novel beyond tooth-shift + apodization |
| `phase0_gate_verdicts_2026-07-06.md` | zero-GPU verdicts of the BIC/Kerker/counterdiabatic phase-0 gates |
| `repo_review_refactor_plan_2026-07-11.md` | full-repo audit + refactor proposal (the §10 code-lifecycle rules came from it) |
| `research_overview_briefing_2026-09-13.md` | raw material for a conclusion-first program overview (not a deck) |
| `theory_gate_supercavity_aniso_2026-07-07.md` | zero-GPU gates: single-cavity/supercavity FW + anisotropic cladding |
| `theory_innermost_recycling_2026-07-08.md` | leaky-field energy pattern + innermost-tooth cancellation ceiling |
| `tm_loss_program_phase0_2026-07-05.md` | TM loss program phase 0 (theory + pending results) |
Also `docs/research_overview_deck_2026-09-13/` (deck JSON + built PDF/PPTX/slide PNGs) and standalone figures (`antineedle_comb_design.{fig,mat,png}`, `invdesign_pillar_space.{fig,png}`, `exotic_innermost_shapes_2026-07-08.png`).

**`runners/`**: `README.md` (the two study patterns + every deploy-menu contract) · `archive/README.md` (closed studies by program) · `metal_mirror/README.md` (lateral-reflector "option B") · `tm/PITCH_ALIGNMENT.md` (how to move a resonance onto a target λ by pitch) · `scatterers/COMB_HANDOFF.md` (the cladding post comb) · `inverse_design/STATUS.md` (lumopt v1: infrastructure works, gradient ~0 — dead end).
**`runners/lumopt2_design/`** (14 md): `HANDOFF.md` (3372, programme state/log; its own box redirects to `HANDOFF_2026-09-01.md`) · `HANDOFF_SELF_CONTAINED.md` (1089, "THE ONE TO USE" — method + 191-param vector + code + raw data) · `HANDOFF_2026-09-01.md` (241, d1-generation state + resume commands) · `HANDOFF_2026-08-30.md` (126) · `THEORY.md` (749, the METHOD only — editable source of the self-contained handoff) · `V2_FWHM_PLAN.md` (1386, FWHM-safe re-optimization plan) · `DESIGNS.md` (259, design registry; all σ/FWHM columns VOID) · `AL_COMBINED_DESIGN.md` (76) · `LIT_REVIEW_2026-08-29.md` (57) · `COMPACTION_PLAN.md` (55) · `CHANGES_2026-08-28.md`, `CHANGES_2026-08-28_night.md`, `CHANGES_2026-08-29.md`, `CHANGES_2026-09-01.md`.
**Elsewhere**: `python_tools/Q3DB_PREDICTOR_HANDOFF.md` (self-contained q3db engine handoff) · `python_tools/archive/README.md` · `matlab_plotting/studies/README.md` (the archive convention) · `igum/README.md` (the second cluster) · 23 `FINDINGS.md`/`MANIFEST.md`/`EDITING.md` inside `results_from_athena/`.

**Supporting infra not covered above**: `athena/` (`deploy_athena.sh`, `athena.conf`, `jobs/*.sh`, `scripts/{athena_run.py,athena_run_one.py,build_sweep_list.py}`) and `igum/` (the maintained mirror — any edit to one must be mirrored or explicitly reported); `container/` (`lumerical.def`, `build.sh`, the 5 GB `.sif`); `zeus/` (older PBS path); `.claude/skills/` (13 skills: `add-study, athena-preflight, athena-status, check-result, dispatch-study, fetch-results, lock-target, lumopt2-design, predict-q3db, safe-compact, stop-runs, update-container, work-alone`). `dgx/` was deleted 2026-09-11 — never recreate a third fork. `.gitignore` excludes `*.mat`, `*.fig`, `*.fsp`, `*.h5`, `results*/`, and all image/PDF rasters — figures and result data are regenerated outputs, not source.
---

# Part 4 — The written recipes (`.claude/skills/`)

These 13 files were written for Claude Code, but their **content is plain-markdown standard
operating procedure** and is worth reading by any assistant or human. They are the distilled
"how to do the recurring task" layer that sits between `CLAUDE.md` (rules) and the code.
Total ~2300 lines. Paths are relative to the repo root.

| File | Lines | What it is |
|---|---|---|
| `.claude/skills/lumopt2-design/SKILL.md` | 1169 | **The complete method record of the inverse-design programme** — the single most information-dense file in the repo after `THEORY.md`. How to run, debug, resume and extend the lumopt2 adjoint campaign, with numbered lesson items (the CLAUDE.md rules cite them as "skill item 35", "skill item 42", …) |
| `.claude/skills/dispatch-study/SKILL.md` | 188 | How to dispatch a study correctly: pick the right deploy flag for the study type, run preflight, smoke-test when required, report job ID + expected outputs |
| `.claude/skills/work-alone/SKILL.md` | 157 | Autonomous-session mode: what to decide alone versus park for the user, watcher discipline, the trouble-finder monitor triage table (benign-recovered / degraded-retrying / FATAL), periodic checkpointing |
| `.claude/skills/update-container/SKILL.md` | 148 | Updating the Lumerical version inside the Athena `.sif` **without moving gigabytes over the VPN** — on-Athena sandbox surgery, verified by checksums and a physics canary |
| `.claude/skills/lock-target/SKILL.md` | 132 | The target-locking method: hit an EXACT value (mode width, peak T / −X dB point, resonance λ, max Q at a T budget) using a knob table plus a linearizing-coordinate ladder, with in-study anchors only |
| `.claude/skills/predict-q3db/SKILL.md` | 112 | The predict-then-confirm workflow for the q3db engine — predict, one confirmation run against pre-registered bands, refit |
| `.claude/skills/athena-preflight/SKILL.md` | 84 | Pre-dispatch safety check: license seats, home-disk quota, queue state |
| `.claude/skills/add-study/SKILL.md` | 74 | Creating a new study/runner so it actually appears in the deploy menus and cannot clobber another study |
| `.claude/skills/fetch-results/SKILL.md` | 67 | Downloading finished results, rendering the study's MATLAB plot headlessly, replying with full local paths |
| `.claude/skills/athena-status/SKILL.md` | 58 | One-shot "how is the run doing?" — queue state, latest job-log tail, freshly produced result files |
| `.claude/skills/safe-compact/SKILL.md` | 50 | Checkpointing a working session so a handoff loses nothing: snapshot server job state, persist programme state and next steps, refresh the task list |
| `.claude/skills/check-result/SKILL.md` | 46 | Loading a `result_*.mat` and reporting T, resonance λ, Q and spatial mode width **correctly**, with the in-window / dead-device sanity check (Part 0 §1.2) |
| `.claude/skills/stop-runs/SKILL.md` | 43 | Safely stopping SLURM jobs: resolve the exact IDs, state them back, cancel, verify |

There is also a persistent memory store at
`C:\Users\evyat\.claude\projects\c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes\memory\`
— ~120 one-fact markdown files (~1 MB) indexed by `MEMORY.md`, split into `project_*`
(study state and verdicts), `feedback_*` (how the user wants to be worked with),
`reference_*` (pointers to external material and literature). **Part 1 of this document is a
digest of the `project_*` and `reference_*` files, and Part 0 §4 digests the `feedback_*`
files** — but the originals carry more detail, including the incident narratives, and they are
the place to look when a number in Part 1 needs its provenance.

---

# Part 5 — Current state and a first-hour checklist

## 5.1 The most recent study (finished 2026-09-29, the day this handoff was written)

**Far-field spherical-harmonic (multipole) decomposition of three 20 µm-mode devices at
N = 98/side.** Goal: same mode width, same length, complex far field **at the resonance**, then
the power fraction per vector spherical harmonic (E/M, l, m).

Devices: **(A)** plain TE, corrugation 250 nm, pitch 500 nm, W800 · **(B)** Itai's re-optimized
Nt60 "overshoot" apodization, the job-63722 geometry untouched (pitch 491.06 nm) · **(C)** plain
TM, corrugation 325 nm, pitch 516.83 nm, W800.

- Runner `runners/sweeps/farfield_sph_20um.py` (has a `SMOKE` knob at the top).
- Analysis tool `python_tools/farfield_multipole.py <mat> [--lmax 200] [--csv]` — full sphere
  from the top (+z) and side (+y) monitors, nearest-normal patchwork, mirror parities read from
  the data, Jackson X_lm / n×X_lm projection. Validated against scipy, against five synthetic
  dipoles (100% in l = 1, correct E/M type), Parseval 1.000.
- **Engine change (default-inert, snapshot gate 6/6 identical):**
  `FarFieldConfig.farfield_freq_points` — default 1 keeps the legacy band-centre behaviour; > 1
  makes the far-field monitors record the band and `extract_farfield` project at the recorded
  point nearest `resonance_wavelength_nm`. Reason: **every previously stored `*_ff.mat` was
  projected at the band centre** — one stored TE example was 41% of a linewidth off resonance.
- Jobs: **164883** smoke FAIL (`apply_monitor_overrides` in `sim_helpers` reset the far-field
  monitors to 1 point — fixed; 2D field planes also produced a 197 MB `.mat` at N=10, so
  `record_2d_fields` is OFF in this study) → **164891** smoke PASS → **164893** round A, 6 tasks.
- **Round A results (all 6 MEASURED, in `results_from_athena/farfield_sph_20um/`):**
  the TE far-field box **converged at y/z 6.8/6.81 µm** (versus 12/12.8: every harmonic within
  0.8 points, T 0.912 both). Widths at N = 98: TE plain **19.13 µm**, overshoot **19.63 µm**,
  TM **19.18 µm** (end-truncation, not the intended 20). The overshoot row reproduced job 63722
  exactly (λ 1559.867 nm, T 0.973, Q 7696, 19.63 µm). Far field within 1–5% of a linewidth of
  resonance in every row.
  **Content:** TE plain is low-order (l ≤ 5 carries 93%): E(3,±3) 19.4%, M(1,0) 13.1%,
  E(4,±4) 12.3%, M(2,±1) 12.0%, E(2,±2) 11.3%. TM (l ≤ 5 = 80%): M(2,±2) 18.1%, M(3,±3) 11.1%,
  E(3,0) 7.6%, E(1,0) 7.1%. The **overshoot apodization is HIGH-order** (mean l ≈ 16, 49% of the
  power above l = 12, a sectoral E(l,l) chain out to l ≈ 22): M(2,±1) 5.9%, E(8,±8) 3.5%,
  E(4,±2) 3.3%, M(1,0) 3.2%.
- Figures + plot script: `matlab_plotting/studies/plot_farfield_sph_20um.m`.

Nothing was running after that. The inverse-design programme was deliberately idle (its two
lanes were cancelled 2026-09-01 *after* their state was fetched), with three local fixes gated
but **undeployed** — see `runners/lumopt2_design/HANDOFF_2026-09-01.md`.

## 5.2 Git state

Branch `add-claude-rules-skills` at `9b8de59`, with a **large uncommitted working tree** (engine
fixes, gates, the `BEST_D1_T9676` design, docs, the deleted `dgx/` fork). Committing and pushing
are permanently user-gated in this project — do not commit or push without an explicit request,
and never `git add` generated figures or result data.

## 5.3 First-hour checklist for a new assistant

1. Read Part 0 of this document, then `CLAUDE.md` (the canonical rule source), then
   `README.md` §"Configuration" and `runners/README.md`.
2. If the task touches the inverse-design programme: read
   `runners/lumopt2_design/THEORY.md` (method) and `HANDOFF_2026-09-01.md` (state) **before**
   reasoning about gradients or running anything.
3. If the task is "what does a longer/shorter device give" or "hit −X dB at W µm":
   use `python_tools/predict_q3db.py` (Part 1 §4) — predict, then ONE confirmation run against
   pre-registered bands, then refit. Do not build a tuning ladder.
4. Before any dispatch: ask **which cluster** (a plain one-line "Athena or IGUM?") unless the
   user already named one; check what already exists in `results_from_athena/` /
   `results_from_igum/`; check license seats and the queue; state what the run decides.
5. Never end a dispatch turn without the job ID and task count.
6. When you learn something the next session would otherwise re-learn, write it into
   `CLAUDE.md` or the relevant handoff **in the same session**.

## 5.4 The five mistakes most likely to be repeated

1. Taking the resonance from `max(T)` — it is in the passband, not at the defect.
2. Confusing `spectral_fwhm_nm` (→ Q) with `fwhm_m` (the acoustic width spec), or forgetting
   that `spectral_fwhm_nm` is stored negative.
3. Re-running a point that already exists somewhere, usually justified as "I couldn't verify
   the numerics were identical". Go read the provenance instead.
4. Dispatching a long job with no incremental persistence, then losing hours to a REQUEUE.
5. Debugging an API/numerics/crash question on the full device at 45–70 min a rung instead of
   on an empty box at seconds a rung.
