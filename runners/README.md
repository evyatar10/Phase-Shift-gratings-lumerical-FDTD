# runners/ — how studies are organized

Every simulation study in this project lives under `runners/`. There are two
deliberate patterns; both are current, and which one you use depends on the
kind of work:

## Pattern 1 — quick studies (`single/`, `sweeps/`)

One small file per study, driven by the shared core modules.

- **`single/`** — one FDTD run per invocation. Any file here with a top-level
  `run` callable (typically `run = run_single_sim`) appears in the Athena
  deploy "Single" menu. `run_simulation.py` is the canonical pipeline;
  `run_experiment.py` runs `ExperimentCard` lists.
- **`tm/`** — TM-polarization single-run studies, kept separate from `single/`
  so the growing TM work doesn't crowd it. Same contract as `single/` (top-level
  `run` callable, `_`-prefixed/`IS_HELPER=True` files skipped) and dispatched
  the same way (sequential OPTION=2 job), but it has its own deploy-menu entry
  ("8) TM studies") and its own auto-discovery dir in `athena_run.py`. The
  runners share one single-scan step in `_tm_vs_te_common.py` (one predefined
  wide window per polarization — no scout/refine two-step):
  - **`run_te.py`** / **`run_tm.py`** — a single wide scan for one polarization,
    so you can run TE and TM separately.
  - **`run_tm_vs_te.py`** — TE-vs-TM comparison: one wide scan per polarization
    (2 sims) + a comparison summary. Run sequentially in one job, or in parallel
    as a 2-task GPU array with `--pol-array` (task 0 = TE, task 1 = TM); for the
    array path, stitch the summary after download with
    `python -m runners.tm.run_tm_vs_te --stitch <results_dir>`.
  - **`calibrate_neff.py`** (`IS_HELPER`, local-only) — FDE n_eff(λ) for TE & TM
    (grating-averaged, constant indices), anchored to the FDTD result, to compute
    the TM pitch that re-centers TM on λ_TE. Run locally (needs a MODE license).
  - **`tm_match_periods_bisect.py`** / **`tm_match_pitch_bisect.py`** /
    **`tm_match_corrugation_bisect.py`** — self-contained integer-bisection /
    secant search drivers: TM period count to match TE@80 peak T; pitch to hit a
    target resonance λ; corrugation to match the TE spatial mode width. Each runs
    a short FDTD sequence per iteration (dispatch like any TM runner).
  - **`tm_wide_mode_corr.py`** — secant search on corrugation for a target
    spatial mode FWHM (the wide-mode H200 devices).
  - **`tm_mode_loss.py`** (`IS_HELPER`, local-only) — FDE modal absorption loss
    (dB/cm), classifying TE/TM by E-power (the FDE "TE fraction" label is
    inverted for this rotated geometry).
  - **`tm_apod_pitch518_rest.py`** — recovery runner for 4 failed points of the
    (archived) pitch-518.3 apodization sweep.

  All runners write to the study's standard `layouts/` (`.fsp`) + `results/`
  (`.mat`) folders and nothing else — the comparison summary is
  `result_tm_vs_te_summary_N<n>.mat` in `results/`. No plots are written; plot the
  spectra in MATLAB from the `result_*.mat`.

  Polarization itself is a one-line `cfg.source.polarization = "TM"`; a plain TM
  single run can also go through `single/`'s `run_simulation` — `tm/` is for the
  TM-specific studies.
- **`sweeps/`** — declarative parameter sweeps. Each file defines a top-level
  `SPEC = SweepSpec(...)` listing only the varying fields; the engine in
  [`sweep_spec.py`](sweeps/sweep_spec.py) expands the cartesian (or zipped)
  product and runs locally or as a SLURM array on Athena.

To add a quick study: copy the closest existing file in the directory, edit
the `SPEC`/config lines, done. It is auto-discovered by the deploy menus.

## Pattern 2 — optimization projects (one directory per method)

Bigger, iterative work gets its own directory with a standard internal layout:

| Directory | Method | Engine entry point |
|---|---|---|
| `inverse_design/` | lumopt adjoint (L-BFGS-B) | `run_inverse_design()` |
| `gradient_free_design/` | Python-driven Lumerical PSO | `run_gradient_free_design()` |
| `fd_gradient_design/` | scipy L-BFGS-B/Powell + finite-difference gradients | `run_fd_gradient_design()` |
| `lumerical_native_optimization/` | Lumerical native `addsweep("Optimization")` PSO | `run_lumerical_native()` |

Standard files inside each:

- **`<dirname>.py`** — the engine: the `*Spec` dataclass plus the full
  optimization loop. Excluded from deploy menus by filename.
- **`optimize_transmission.py`** — the production study: `BASE` config +
  `SPEC` instance (n_periods=80).
- **`smoke_test.py`** — same wiring at n_periods=20 with a tiny iteration
  budget; run this on Athena first to catch wiring bugs in ~15-30 min before
  burning a production slot.
- **`test_geometry.py`** (where present) — geometry math validation.
  `python -m runners.inverse_design.test_geometry` runs locally with NO
  Lumerical license; `runners/gradient_free_design/test_geometry.py` opens a
  lumapi session and needs a license.
- **`plot_run.py`** (where present) — convergence/spectrum plots for a
  finished run, built on `visualization/plot_optimization.py`.

Related project-style directories:

- **`experiment_comparison/`** — simulation vs fabricated-device (IT11)
  comparison: `device_names.csv` → `it11_card_builder.py` → card lists
  (`it11_devices*.py`, `two_resonators.py`), overlay plots via
  `compare_with_experiment.py`.
- **`visualization/`** — shared plotting helpers for optimization runs.

## Shared modules map

- **`optimization_common.py`** (runners/ root) —
  `make_optimization_base(n_periods)`: the base `SimulationConfig` shared by
  all 8 optimization spec files (20 = smoke, 80 = production). If you change
  it, you change every optimization study at once — that is the point.
- **`inverse_design/inverse_design.py`** — de-facto geometry library used by
  all four methods: `regular_grating_start`, `params_to_kwargs`,
  `measure_baseline`, `freed_region_x_bounds`, `freed_region_named_rects`,
  `_build_static_skeleton`.
- **`gradient_free_design/gradient_free_design.py`** — the parametric-FSP
  helpers `_build_parametric_fsp`, `_set_freed_group_params`,
  `_make_fom_analysis_script` are shared by fd_gradient_design and
  lumerical_native_optimization (public aliases without the underscore are
  exported at the bottom of the file). Don't rename/move them without
  updating the importers.

A number of one-off study files (check_gradient_test*, scan_cavity_width,
optimize_transmission_outer/_corrected, local_smoke, the sweeps) still build
their own `BASE = SimulationConfig()` with study-specific values — that is
intentional; only the 8 standard spec files use `make_optimization_base`.

## The deploy-menu contract (READ BEFORE RENAMING/ADDING FILES)

`athena/deploy_athena.sh` and `dgx/deploy_dgx.sh` discover studies by
scanning this tree. The rules, verified against the scripts:

1. **Directory names are hardcoded** in the deploy menus. Never rename
   `single/`, `tm/`, `sweeps/`, or the four optimization directories. A new
   single-run-style category (like `tm/`) needs three edits: the picker block +
   menu entry in `deploy_athena.sh`, and a `_AUTO_DIRS` entry in BOTH
   `athena/scripts/athena_run.py` and `dgx/scripts/athena_run.py`.
2. **Single / TM menus** = every `runners/single/*.py` (resp. `runners/tm/*.py`)
   with a top-level `run` callable (`def run(` or `run = ...` at column 0);
   `_`-prefixed files and `IS_HELPER=True` modules are skipped.
3. **Study menus** = every file in the family directory containing the
   literal text `SPEC =` anywhere — it is an unanchored grep, so even a
   docstring or comment containing `SPEC =` puts a file in the menu. The
   engine file is excluded by filename only.
4. **Shared helper modules** must avoid those triggers: put them at the
   `runners/` root (never scanned — `optimization_common.py` lives there for
   exactly this reason), or make sure they contain neither a top-level `run`
   nor the literal text `SPEC =`.
5. **rsync `--delete`**: deploys mirror the whole `runners/` tree to the
   cluster. A file deleted/moved locally disappears from the server's
   `project/runners/` copy on the next deploy (results under
   `~/bragg_sim_athena/results/` are untouched — only synced code).

## Typical workflows

```bash
# Quick single sim (local build, Athena run)
bash athena/deploy_athena.sh --option2 --run=run_simulation

# Parameter sweep as a SLURM array
bash athena/deploy_athena.sh --spec=runners.sweeps.number_of_periods

# Experiment-card batch
bash athena/deploy_athena.sh --cards=runners.experiment_comparison.it11_devices

# Optimization: ALWAYS smoke first, then production
bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.smoke_test
bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission
# Other families: --inverse-design= / --lumerical-native= / --fd-gradient-design=
```

## archive/ — closed studies

When a study is declared closed, its one-off runner/analysis files move to
`runners/archive/` (and its one-off MATLAB plots to `matlab_plotting/studies/`),
unedited. The deploy menus never scan `archive/`, so archived files leave the
menus but stay runnable via their new module path (see
[archive/README.md](archive/README.md) for the per-program subfolders:
side_by_side, scatterers, cavity_shapes, loss_program, bic_kerker_trench,
apodization, convergence).

The live `sweeps/` directory holds only: the engine (`sweep_spec.py`), the
shared anchored-TM base (`_tm_base.py` — imported by both live and archived
TM sweeps; don't move or rename it), the four canonical example sweeps
documented above, the shift-summary pair (`tm_te_shift.py`,
`tm_shift_p518.py`, `plot_tm_te_shift.py` — referenced by
`athena/jobs/run_shift_summary.sh` and the deploy scripts), and currently
active studies.

## Legacy files (kept runnable, superseded)

- `single/run_simple_bragg.py` — self-contained uniform-Bragg runner
  (no cavity/apodization); predates the SimulationConfig pipeline.
- `single/verify_pso_best.py` — one-off verification of an early PSO result;
  the gradient_free_design pipeline now does its own post-opt verification.
- `sweeps/optimize_innermost_shift.py` — Brent's-method shift optimizer;
  superseded by the optimization directories (it never appears in the sweeps
  deploy menu — it defines no `SPEC`).
