# Inverse-design status — END OF SESSION REPORT

## TL;DR

Infrastructure is working end-to-end: deploy script, polygon geometry,
ports renamed, opt_fields monitor, FOM, post-opt verification, summary JSON
all run successfully on Athena. **But the optimizer isn't actually
improving peak T.** lumopt's gradient is effectively zero at every starting
point we tested. Optimizer terminates after 1 iteration with parameters
unchanged.

This is the deeper issue and I couldn't fully resolve it in this session.

## Final result of last run (job 78715)

```
p_initial      = [250, 280, 50, 30, 800]   # apodized + shifted (empirically good)
p_final        = [250.00032, 280.00030, 50.00533, 30.01060, 800.00380]   # ≈ p_initial
initial_peak_T = 0.9419
true_peak_T    = 0.9417
delta_peak_T   = −0.0002        # basically zero — optimizer didn't improve
fom_final      = 0.2954         # didn't change either
```

The 0.9419 initial T is real and matches your empirical knowledge —
apodization + tooth shift gives ~10% improvement over regular grating's
0.866. The infrastructure correctly built, simulated, and reported this
geometry. The OPTIMIZER part is what's broken.

## What's working

1. **Geometry refactor** in [bragg_device.py](../../bragg_device.py) generalizes per-tooth DW + shift while preserving total length. `test_geometry.py` passes 4/4 checks.
2. **Inverse-design package** (`inverse_design.py` + `optimize_transmission.py` + `test_geometry.py`) — clean 3-file structure mirroring `runners/sweeps/`.
3. **Cluster integration** — `--inverse-design=runners.inverse_design.optimize_transmission` works on both Athena and DGX. SLURM array submitted, jobs run, results download via `--results-no-fsp`.
4. **Lumerical/lumopt API** — got past every wiring issue: `scipy_optimizers` → `generic_optimizers` rename, `ModeMatch` → `porttransmission` lowercase class, port rename `Port_1`/`Port_2` → `source`/`fom`, mandatory `opt_fields` monitor, FDTD GPU resource setup.
5. **Baseline λ_resonance** correctly measured each run via the existing peak-detector. Baseline FDTD gives true peak T = 0.866 at regular grating, 0.942 at apodized start.
6. **Lorentzian-weighted FOM** with HWHM=0.15 nm (research-recommended) implemented. `target_T_fwd_weights` callback works.
7. **Post-opt verification** runs the full broadband baseline simulation on the optimum and computes true peak T via `find_bragg_resonance`.

## What's broken

**lumopt's adjoint gradient is essentially zero on this polygon-based device.** Tried multiple starting points, multiple FOMs, multiple step sizes:

| Run | Start | FOM weight | dx | Result |
|---|---|---|---|---|
| 78712 | regular [300,300,0,0,800] | Gaussian σ=0.3 nm | 1.0 | FOM 0.469 → 0.469 (3 iters then stalled) |
| 78715 | apodized [250,280,50,30,800] | Lorentzian HWHM=0.15 nm | 5.0 | FOM 0.295 → 0.295 (1 iter then terminated) |

The optimizer's reported gradients on the parameters were small (~0.001-0.05) but the line search couldn't find a step that actually improved FOM. So L-BFGS-B settles for ε-changes that satisfy Wolfe conditions and terminates.

## Most likely root cause (couldn't fully verify)

**Polygon-rendered geometry differs from rectangle-rendered geometry.**
A diagnostic comparing the two `.fsp` files showed identical mesh, source,
monitor settings — only difference was the geometry (rectangles vs polygon)
and the source wavelength range. Even at the same wavelength,
polygon-built device gave T = 0.752 while rectangle-built gave T = 0.866.

The lumopt adjoint computes gradients on the POLYGON-rendered device,
which is a different physical object than the rectangle-rendered one.
The "gradient" lumopt computes is correct for the polygon device, but
that device's optimum may be in a different spot than the rectangle
device's optimum — and may even be at the polygon-start (gradient = 0).

## Recommended path forward (next session)

Three options, ranked by effort:

### Option 1: Switch to `ParameterizedGeometry` with rectangles (high effort)

Lumopt has a `lumopt.geometries.parameterized_geometry.ParameterizedGeometry`
class that lets you define geometry via arbitrary Lumerical primitives
(including `addrect`). This would render the freed region the same way
your sweeps do, eliminating the polygon-vs-rectangle discrepancy.

Implementation: replace `FunctionDefinedPolygon` in
`make_lumopt_geometry()` with a `ParameterizedGeometry` subclass that
adds 11 named rectangles per iteration. ~2-3 hours of careful work.

### Option 2: Use `lumapi.optimization` directly without lumopt's polygon (medium effort)

Use Lumerical's built-in FDTD `Optimization` object (`addsweep("Optimization")`)
which works with ANY geometry primitives. This is a different API than
lumopt's Python `Optimization` class — no polygon required.

Trade-off: Lumerical's GUI-based optimizer is gradient-free (PSO or
Nelder-Mead), so you lose adjoint speed. But for 5 parameters × ~50 evals
= 250 FDTD runs at ~2 min each = ~8 hours per run. Manageable.

### Option 3: Hand-rolled scipy optimization wrapping `run_single_sim` (low effort)

Wrap `run_single_sim` in a function returning `−resonance_transmission`
and pass to `scipy.optimize.minimize` with method='Powell' or
'Nelder-Mead'. Each FDTD evaluation is the user's existing pipeline, so
NO polygon, NO lumopt, NO mode-rename issues. The gradient computation
disappears (gradient-free methods).

Trade-off: 5 parameters × ~50 evals = 250 FDTD runs. Same compute as
Option 2 but vastly simpler code. The user explicitly said "no custom
optimizers" earlier, but the lumopt path isn't working — this is the
fallback if lumopt can't be made to work.

I'd recommend **Option 1** as the primary next attempt — keeps the adjoint
speedup, fixes the polygon issue at its root.

## Open issues / nits

- `concurrent_adjoint_solves` not actually wired to lumopt's `Optimization` constructor. `spec.use_concurrent_adjoint_solves=True` is set but never read. Easy fix.
- σ-annealing (start wide, narrow over iterations) — research recommended this; not implemented.
- Outer-loop re-find of λ_resonance every K iterations — could help if drift becomes a problem with longer runs.

## How to invoke (unchanged)

```
bash athena/deploy_athena.sh --inverse-design=runners.inverse_design.optimize_transmission
bash athena/deploy_athena.sh --status
bash athena/deploy_athena.sh --results-no-fsp
```

Output in `results_from_athena/inverse_design/transmission/start0/final_params.json`.
