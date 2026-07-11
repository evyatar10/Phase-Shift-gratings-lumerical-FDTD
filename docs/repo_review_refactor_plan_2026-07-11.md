# Repo review & refactoring plan — 2026-07-11

Full-repo audit (4 parallel sweeps: core modules, runners/, MATLAB, deploy infra).
Nothing has been changed yet — this is the proposal. Each stage lists its own
verification recipe so code logic is provably preserved.

**Confidence labels used below:** VERIFIED = checked directly in this audit;
LIKELY = strong grep/diff evidence but not exhaustively proven; UNSURE = flagged
as uncertain by the audit.

---

## 1. What the audit found

### 1.1 The big picture

The project is **not badly engineered — it is a research lab notebook that grew**.
The core pipeline (`simulation_config.py` → `bragg_device.py` → `post_processing.py`)
is documented and structured; the two READMEs are accurate about the parts they
cover. The mess is concentrated in three places:

1. **Spent one-off study scripts** accumulate forever in live directories
   (~48 of 59 MATLAB plot scripts, ~55 sweep files, the whole `side_by_side/` tree).
2. **Three forked copies of the deploy infrastructure** (athena/igum/dgx) that
   have genuinely drifted apart.
3. **A few god-objects** in the core (`bragg_device.__init__` = 71 kwargs / ~515
   lines; `_add_bragg_core` ≈ 362 lines; `generate_file_tag` ≈ 215 lines;
   `deploy_athena.sh` = 1349 lines, 26 flags, 3 functions).

### 1.2 Concrete inventory of issues

**Dead / stray files (VERIFIED unreferenced unless noted):**
- Root: `check_all_tm_fields.m`, `check_fields_temp.m` — tracked, zero references.
- Root: `CUsersevyatAppDataLocalTemppowell_result.json` — untracked garbage (a
  Windows temp path collapsed into a filename by a mis-redirect).
- `legacy/post_process_old.py` — tracked, zero references.
- `debug_fsp_compare/baseline.fsp`, `lumopt_forward.fsp` — **binary .fsp files
  are git-tracked** (`.fsp` is missing from `.gitignore`); also
  `test_singleλ.py` has a non-ASCII filename that renders mangled in git.
- `runners/side_by_side/__pycache__/` — orphaned bytecode whose source files no
  longer exist.
- `sweeps/view_tm_match_interactive.py`, `sweeps/view_tm_match_plotly.py`,
  `sweeps/plot_tm_periods_match_te.py`, `experiment_comparison/pull_by_subname.py`
  — zero external references (LIKELY dead; local viewers/utilities).
- `matlab_plotting/plot_air_trench_peakT_regular.m` vs
  `plot_air_trench_regular_peakT.m` — two names for the same job-119488 plot;
  at most one is live (VERIFIED same target).
- Tracked figure artifacts in `matlab_plotting/`: `plot_compare_8_devices.fig`,
  `plot_te300_vs_tm400_match.fig/.png` — violate CLAUDE.md §7 (predate the rule).

**Stale docs / config (VERIFIED):**
- Root `README.md` lists `matlab_analysis/calculate_profile.m` — the file does
  not exist.
- Root `README.md` and `runners/README.md` do not mention: `runners/side_by_side/`
  (23 files), `runners/tm/` bisection runners (4 large files), `tm_mode_loss.py`,
  ~10 undocumented `single/` runners, all ~48 one-off MATLAB scripts,
  `python_tools/phase0_*` family.
- `.claude/settings.json` `additionalDirectories` points to `hpc\scripts`, which
  does not exist.
- `.cursor/plans/parameter_clarity_refactor_*.plan.md` — the old refactor plan is
  mostly already implemented (SimulationConfig groups exist); stale and misleading.

**Cross-cluster drift (VERIFIED by diff):**
- `dgx/scripts/athena_run.py` is missing two athena features (the
  `run_mesh_convergence` alias and the `STUDY_DIR_NAME` override block).
- The `setresource("FDTD", processes=1)` fix exists in two different states
  (commented-out on dgx, removed on athena) — same knowledge, forked code.
- `dgx/deploy_dgx.sh` lacks 4 of athena's 26 flags; igum has full parity.
- Job scripts differ by 28–153 lines per file across clusters; some divergence is
  legitimately platform-specific (container vs native, NVML trampoline), but the
  structure is a hand-maintained triple fork.
- Note: memory says the DGX GPU path is **silently broken** with Lumerical 2026R1
  anyway (`project_dgx_fdtd_gpu_broken.md`).

**Core-module hotspots (VERIFIED):**
- `bragg_device.py` (1470 L): `__init__` L18–L533 with 71 kwargs;
  `_add_bragg_core` L772–L1134; `_add_source_and_monitors` ~197 L.
- `simulation_config.py`: `to_device_kwargs` (~120 L) mirrors the 71-kwarg
  surface — every new parameter is added in ≥2 places.
- `sim_helpers.py`: `generate_file_tag` L299–L514.
- `experiment_card.py` lazily imports `runners.single.run_simulation` (root →
  runners inverted dependency; harmless because lazy, but worth knowing).
- No TODO/FIXME markers anywhere in core; sampled "suspicious" config fields are
  all referenced (no dead config fields found in the sample; not exhaustive).

**Fragile couplings in runners/ (VERIFIED):**
- 7 `sweeps/*` files import the underscore-private `tm/_tm_vs_te_common`;
  `tm/tm_apod_pitch518_rest.py` imports back into `sweeps/` (bidirectional
  coupling).
- `single/tm_baseline_accurate.py` imports `BASE` from
  `gradient_free_design/optimize_transmission_tm.py` (a runner depending on a
  specific study file's module constant).
- All four optimization methods import from `inverse_design/inverse_design.py`
  (1453 L) and `gradient_free_design/gradient_free_design.py` — renaming either
  breaks three sibling dirs (README already warns).

**Good news (VERIFIED):**
- The `plot_transmission.m` Q-factor bug from memory is FIXED in-code
  (outward-walk FWHM + interpolation, documented `% BUGFIX:` comment).
- No MATLAB script points at a deleted results directory.
- No tracked `.png/.mat/.h5/.fig` outside the two legacy figure artifacts above.
- `.gitignore` otherwise has good coverage.

---

## 2. Refactoring plan (staged, safest-first)

Design principle: **archive, don't rewrite, spent studies** — one-off scripts are
the lab notebook; rewriting them risks changing recorded science for zero benefit.
Refactor only the living engines. Every stage is independently shippable.

### Stage 0 — hygiene, zero behavior risk (~30 min)

1. Add `*.fsp` to `.gitignore`; `git rm --cached` the two tracked `.fsp` binaries
   (files stay on disk). Same for the 3 tracked `.fig/.png` artifacts.
2. Delete (with permission prompt): the mangled `CUsersevyat...json`,
   `check_fields_temp.m`, `check_all_tm_fields.m`, `legacy/post_process_old.py`,
   the orphaned `side_by_side/__pycache__/`, the stale `.cursor` plan.
3. Fix `README.md`: remove the nonexistent `calculate_profile.m` entry.
4. Fix `.claude/settings.json`: remove the dead `hpc\scripts` additionalDirectory.
5. Rename `debug_fsp_compare/test_singleλ.py` → `test_single_lambda.py`.

*Verify:* `git status` clean of surprises; `python -m compileall .` still passes;
deploy menus unchanged (none of these files is menu-discovered).

### Stage 1 — archive spent studies (organizational, no code edits)

1. Create `runners/archive/` and `matlab_plotting/studies/` (or `archive/`).
2. Move the closed-study one-offs there **without editing their contents**:
   - MATLAB: the ~48 job-specific `plot_*.m` / `make_*.m` scripts (keep the 11
     engine scripts + `plot_prefs.mat` where they are, because
     `.claude/skills/fetch-results` cd's into `matlab_plotting/`).
   - Python: `side_by_side/` tree (study CLOSED per memory), spent sweep families
     the user confirms closed (scatterers, cavity shapes, …), the
     `check_gradient_test_*` variants, `python_tools/phase0_*` (8 files).
3. Add the archive dirs to `startup.m` path so old MATLAB scripts remain runnable.
4. One-paragraph README in each archive dir: "closed studies, kept runnable,
   see memory/FINDINGS for conclusions".

*Why safe:* deploy menus scan hardcoded dirs (`single/`, `tm/`, `sweeps/`, the 4
optimization dirs) — files moved OUT of them simply leave the menu; nothing else
imports the archived files (verified per-file above; the `side_by_side` tree is
imported by nothing live).
*Caveats to check per file before moving:* (a) the 7 sweeps importing
`tm/_tm_vs_te_common` must stay with their helper or move together; (b) rsync
`--delete` will remove moved files from the server's `project/` copy on next
deploy — results are untouched, but don't do this while an array job is PENDING
(sweep_list bounds-check incident, CLAUDE.md §6).
*Verify:* before/after diff of the deploy menu listings (run the menu-generation
greps from `deploy_athena.sh` on both trees and compare); `python -m compileall`;
`checkcode` on moved MATLAB files.

### Stage 2 — documentation truth-sync (~1 h)

1. `runners/README.md`: add `side_by_side` (archived) note, document the `tm/`
   bisection runners and `tm_mode_loss.py`, document the undocumented `single/`
   runners (or archive them in Stage 1).
2. Root `README.md`: add `python_tools/` actives, `docs/` convention, archive
   convention.
3. `FILE_NAMING.md`: already accurate; add the newer tokens if any are missing.

### Stage 3 — deploy infra consolidation (medium risk, big maintenance win)

Recommendation ordered by cost/benefit:
1. **Declare dgx/ legacy-frozen** (it's broken with 2026R1 per memory): add a
   README banner "not maintained — reference only", stop propagating fixes.
   Alternatively archive the whole dir. This halves the fork-maintenance surface
   with zero code risk.
2. **Port the two missing athena features to igum** (`athena_run.py` is only ~18
   deliberate lines apart — keep igum current since it's the live second target).
3. Extract the byte-identical `build_sweep_list.py` to one shared location only
   if the deploy scripts are touched anyway — otherwise leave (identical copies
   that never diverge are cheap).
4. `deploy_athena.sh` (1349 L): do NOT rewrite wholesale. Incremental
   function-extraction only when a section is next edited (submission paths →
   functions). A working 1349-line dispatch script with 35 sessions of scar
   tissue is an asset; a fresh rewrite is how job-dispatch bugs get reintroduced.

*Verify:* `--status`, `--license-probe`, menu rendering, and one `--option2`
smoke dispatch after any deploy-script edit; igum edits verified with its
`--upload-only` + menu listing (SLURM assoc still blocked per memory).

### Stage 4 — core module refactor (highest value, do only with appetite)

1. **`bragg_device.PiShiftBraggFDTD`**: keep the 71-kwarg `__init__` signature
   (every runner calls it), but mechanically split the body: move the ~19
   banner-separated sections of `__init__` into private `_init_*` helpers and
   `_add_bragg_core` into per-feature helpers (teeth loop, cavity, shapes,
   scatterer hooks). Pure extraction, no signature changes.
2. Longer term: let `__init__` accept `cfg: SimulationConfig` directly and make
   `to_device_kwargs` the compatibility shim, so new parameters are added in ONE
   place. This is the real fix for the 2-place parameter plumbing but touches
   every call site's mental model — needs a dedicated session.
3. `generate_file_tag`: table-driven rewrite (list of (condition, token) rules).

*Verify (this is the strong part):* the repo already contains the right tool —
`debug_fsp_compare/diff_fsp.py`. Recipe: for N representative configs (TE
baseline, TM 516.83/400, apodized, shifted, scatterer, simple-bragg), build
`save_fsp` locally BEFORE and AFTER the refactor and diff the `.fsp` scenes;
plus assert `generate_file_tag` output is byte-identical across a grid of
configs (quick pytest). Identical scenes + identical tags ⇒ logic preserved.

---

## 3. Recommended CLAUDE.md rule additions

### 3.1 Honesty / calibration rule (requested) — proposed §9

```markdown
## 9. Honesty & calibration (overrides style, speed, and optimism)

- **Report what happened, not what was hoped.** Failed test, undispatched job,
  skipped step, partial download, empty result — state it first and prominently,
  before any summary. "Done" is only for things actually done and checked.
- **Label every quantitative claim** as one of: MEASURED (read from a named file
  this session — cite the file), DERIVED (computed from measured values — show
  from what), or EXPECTED (theory/memory/estimate — say so). Never state numbers
  from a file that wasn't opened this session.
- **No overstatement.** Near-noise-floor effects (§2), single-point results, and
  unconverged optimizations are "candidate"/"preliminary" — never "confirmed",
  "proven", "best", or "significant" until the §2 sanity checks pass. State the
  uncertainty with the claim, not after being asked.
- **"I didn't check" beats a plausible guess.** A confident wrong answer costs
  GPU-hours; "unsure, let me verify" costs a minute. When memory and current
  code disagree, the code wins and the memory gets corrected.
- **Push back on wrong premises.** If the user's assumption contradicts the data,
  say so directly instead of building on it.
```

**Honest caveat about this rule (important):** a CLAUDE.md rule biases behavior
strongly but cannot *guarantee* truthfulness — no instruction can. The reliable
part is the *mechanism* it mandates: forcing MEASURED/DERIVED/EXPECTED labels and
file citations makes claims checkable, and the existing §2 sanity gates catch the
worst class (confident garbage from dead devices). Rules + verifiable-claim
format + hard gates is the honest maximum available.

### 3.2 Other rule updates recommended

- **§7 extension:** add `*.fsp` to the never-commit list (two were tracked).
- **New: study lifecycle.** "When a study is declared closed, its one-off runner
  and plot scripts move to `runners/archive/` / `matlab_plotting/studies/` in the
  same session; every new one-off plot script's header states the study dir and
  job ID." (This is the anti-mess rule — the audit shows ~90 accumulated one-offs.)
- **New: cluster mirroring.** "athena/ and igum/ scripts are a maintained pair:
  any edit to `athena/scripts/*` or `athena/jobs/*` is either mirrored to igum/
  in the same change or explicitly reported as not mirrored. dgx/ is frozen
  legacy — do not edit." (Prevents the observed drift class.)
- **New: memory correctness.** "When an audit shows a memory/CLAUDE.md claim is
  stale (e.g. a bug since fixed), update the memory in the same session." (Applied
  today to the Q-factor memory.)
- **§8 minor:** point to the stale-doc failure mode: "when adding/removing a
  documented file, update README in the same change" — the `calculate_profile.m`
  ghost entry shows the cost.

---

## 4. Recommended skills (new / updated)

Existing 7 project skills (add-study, athena-preflight, athena-status,
check-result, dispatch-study, fetch-results, stop-runs) cover the dispatch
lifecycle well. Gaps, in priority order:

1. **`close-study`** (new) — the highest-value addition given the audit. When the
   user says a study is done/closed: move its runner+plot one-offs to the archive
   dirs, confirm the results location, write/update the memory file, verify
   deploy menus unchanged. This is the mechanism that keeps the repo from
   re-accumulating the current mess.
2. **`new-plot`** (new) — scaffold a MATLAB study plot from a template with §8
   conventions baked in: title = dimensions + resonance λ + peak T, Top/Side view
   naming, `'Interpreter','none'`, `.fig`+PNG export, and **ending the reply with
   full absolute paths** (the thing that has been asked 21 times). Reduces the
   6-variant apodization-plot copy-paste class.
3. **`repo-health`** (new, optional) — re-run today's checks quarterly: tracked
   artifacts, orphaned files, README-vs-tree drift, athena↔igum diff, stale
   settings entries.
4. **Update `dispatch-study`** — add one line: "if the change touched
   athena/scripts or athena/jobs, check igum mirror status before dispatching."
5. **Update `fetch-results`** — if Stage 1 moves MATLAB one-offs into
   `matlab_plotting/studies/`, add that dir to its script-lookup path.

---

## 5. Not recommended (explicitly)

- Rewriting `deploy_athena.sh` from scratch, unifying the 3 clusters into one
  parameterized script, introducing a plugin/registry system for studies, or
  converting one-off studies into a generic framework. All are
  scope-creep-shaped (CLAUDE.md §8 / `feedback_avoid_overcomplication`): the
  plain-file workflow works, and the risk lands on the job-dispatch path where
  bugs cost GPU-hours.
- Mass-editing the ~48 archived MATLAB scripts to use shared helpers. Add shared
  helpers (`load_result.m`, style helper) for FUTURE scripts via the `new-plot`
  skill instead.
