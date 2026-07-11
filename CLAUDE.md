# CLAUDE.md — Pi-Shift Bragg Grating FDTD

Project rules for Claude Code. These are always-on invariants. They were distilled
from ~35 prior sessions; the incidents behind each rule are real and cost real GPU
hours. Read `README.md` for architecture and `runners/README.md` for the study patterns.

The device is a **pi-shift Bragg grating** (use this term in discussion/writeups).

---

## 1. Where things run

- **Default: run FDTD on Athena**, dispatched via `bash athena/deploy_athena.sh`.
  Athena has good availability and is far faster than the local machine.
- **IGUM (ECE faculty cluster) is a second, coexisting option** — `bash
  igum/deploy_igum.sh`. Native Lumerical (NO containers there), submission needs
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
  - **Any edit to `bragg_device.py` geometry/monitor code:** run
    `python debug_fsp_compare/scene_snapshot.py --out <tmp>` and diff against
    the committed `debug_fsp_compare/snapshots/` references (6 configs spanning
    the builder's code paths; byte-identical = behavior preserved). Regenerate
    the references only when a geometry change is INTENDED, and say so.
  - Parametric/PSO builders: score gen-0 / seed against the known-good baseline; if it
    doesn't match, the builder is broken (use `rebuild_per_particle`).
  - New gradient method: finite-difference `check_gradient` on a tiny problem before
    scaling (hard gate: `vec_error` must be small).
  - MATLAB: `checkcode` lint + headless `exportgraphics` render.
- **Skip** re-verifying known-good baselines and re-linting untouched code. Don't invent
  extra test passes for mechanical edits.

## 6. Server safety

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
  **NO exceptions to the serialize rule** — a DIFFERENT study, "just a 4-task job",
  or a re-deploy of an already-running study's code all rewrite the shared
  `data/sweep_list.txt`, and every pending array task bounds-checks its index against
  that file *at task start*: tasks beyond the new length die instantly with
  "SWEEP_INDEX out of range" while `sacct` can still show early tasks COMPLETED
  (2026-07-02: hole-scan tasks 13–97 killed this way by a 4-task demo deploy; check
  task LOGS, not just states). Recovery: wait for queue-empty, redeploy the clobbered
  study, resubmit the dead range via `--array-tasks=<lo>-<hi>`.
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
- **Silent no-ops.** A license outage makes `fdtd.run()` return instantly with no
  results. If a run finishes implausibly fast / empty, check the license before
  re-dispatching. If a job crashes <30 s right after a config change, suspect **stale
  server code**: restrictive dir perms on remote `project/` can make rsync silently skip
  root `*.py` files (`rsync --inplace` is the known fix — verify the deploy's itemized
  output actually updated the files you edited).
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
