# CLAUDE.md — Pi-Shift Bragg Grating FDTD

Project rules for Claude Code. These are always-on invariants. They were distilled
from ~35 prior sessions; the incidents behind each rule are real and cost real GPU
hours. Read `README.md` for architecture and `runners/README.md` for the study patterns.

The device is a **pi-shift Bragg grating** (use this term in discussion/writeups).

---

## 1. Where things run

- **Default: run FDTD on Athena**, dispatched via `bash athena/deploy_athena.sh`.
  Athena has good availability and is far faster than the local machine.
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

## 5. Verification policy (smoke-test, don't over-test)

History is one-sided: under-testing repeatedly burned GPU hours (dead parametric TM
device ~8 GPU-h; bad-gradient lumopt ~30 GPU-h; `phi=-90` source wasted weeks).
Over-testing never once cost anything. So:

- **Smoke-test before dispatch when the change touches:** (1) device geometry, (2) a new
  builder / parametric scaffold, (3) inverse-design / gradient equations, (4) source or
  boundary-condition setup. Especially for anything **new**.
  - Lumapi: local build-only `save_fsp` (<1 min) + eyeball the geometry.
  - Parametric/PSO builders: score gen-0 / seed against the known-good baseline; if it
    doesn't match, the builder is broken (use `rebuild_per_particle`).
  - New gradient method: finite-difference `check_gradient` on a tiny problem before
    scaling (hard gate: `vec_error` must be small).
  - MATLAB: `checkcode` lint + headless `exportgraphics` render.
- **Skip** re-verifying known-good baselines and re-linting untouched code. Don't invent
  extra test passes for mechanical edits.

## 6. Server safety

- **Concurrency / no clobbering.** Deploy does `rsync --delete` into a *shared*
  `REMOTE_BASE/project/` and writes to a *shared* `results/` + `data/sweep_list.txt`.
  Two chats/jobs deploying at once **overwrite each other's source and outputs** (real
  incidents: `sweep_list.txt` cut 48→14 lines; shared `.h5` filenames raced). Before
  dispatching: **check `--status` / `squeue`**; don't launch a second `--option3` sweep
  while another has pending tasks; ensure per-config unique output filenames
  (`generate_file_tag()`), and **serialize** jobs that share mutable state.
- **Stopping runs is a confirm-first action.** Never blanket `scancel`. Resolve the
  specific job ID from `squeue` first, state it back, and confirm before cancelling.
  After cancel, re-check `squeue` to verify. Treat "stop the run" as needing a job ID,
  not speed.
- **Disk quota.** Home has a ~300 GB quota; rebuild-PSO fills it with `.fsp`+`.h5` and
  then jobs silently hang at container init ("Setting --writable-tmpfs"). If jobs hang
  or quota is near 300 G, **delete `.h5` scratch** (don't keep `.h5` by default).
- **Silent no-ops.** A license outage makes `fdtd.run()` return instantly with no
  results. If a run finishes implausibly fast / empty, check the license
  (`--license-probe` / `lmstat`) before re-dispatching.

## 7. Don't commit artifacts

Figures and data are regenerated outputs, not source. `.gitignore` covers
`*.mat`/`*.fig`/`*.h5`/`results*/` and now image rasters (`*.png` etc.). Don't `git add`
generated figures or result data; if you see them staged, flag it.

## 8. Style

- Keep changes minimal and match surrounding code. Don't propose snapshot/auto-save/
  helper-CLI layers on top of workflows that already work via plain file edits.
- Start optimizers from a known-good baseline (regular grating), not multi-start LHS.
- **Links: give the full path, not just a relative one.** When linking to a file, use
  the full absolute path (e.g. `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\matlab_plotting\plot_transmission.m`)
  in the link target, not a bare relative/local path like `matlab_plotting/plot_transmission.m`.
