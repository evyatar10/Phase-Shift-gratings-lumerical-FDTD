"""
Inverse-design entry point: lumopt continuous-adjoint shape optimization
of the pi-shift Bragg grating's peak transmission.

Workflow per driver (one SLURM array task):
  1. Optionally run a baseline FDTD to determine λ_resonance via
     `sim_helpers.find_bragg_resonance` (skipped if `spec.baseline_lambda_m`
     is preset).
  2. Build a lumopt `Optimization` object: ModeMatch FOM at the single
     baseline wavelength, FunctionDefinedPolygon over the freed region,
     ScipyOptimizers (default L-BFGS-B).
  3. Run for `spec.max_iter` iterations.
  4. After convergence: re-simulate the optimum across the full scan band
     and report the true peak T via `peak_t_diagnostic`.

Per-driver results are written under
    {BASE_SAVE_DIR}/inverse_design/{label}/start{K}/
with `final_params.json` summarizing the trajectory and the true peak T.

Invocation:
  python -m runners.inverse_design.peak_t_adjoint --spec runners.inverse_design.<study> --start 0
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from typing import Optional

import numpy as np

import config as _cfg
from simulation_config import SimulationConfig
from runners.inverse_design.spec import InverseDesignSpec, params_to_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Baseline λ_resonance extraction
# ─────────────────────────────────────────────────────────────────────────────

def measure_baseline_lambda_m(cfg: SimulationConfig) -> float:
    """Run a single FDTD at the cfg's baseline geometry, return λ_resonance.

    Uses the existing `runners.single.run_simulation.run_single_sim` so the
    same monitor / phase-correction / peak-detection pipeline applies.
    """
    from runners.single.run_simulation import run_single_sim
    from sim_helpers import peak_t_diagnostic

    print("[inverse_design] measuring baseline λ_resonance ...")
    result = run_single_sim(cfg)
    wl = np.asarray(result["wl"]) if "wl" in result else np.asarray(result["wavelength"])
    T = np.asarray(result["T"]) if "T" in result else np.asarray(result["transmission"])
    lam_peak, T_peak = peak_t_diagnostic(wl, T)
    print(f"[inverse_design] baseline λ_resonance = {lam_peak * 1e9:.3f} nm  (T = {T_peak:.4f})")
    return float(lam_peak)


# ─────────────────────────────────────────────────────────────────────────────
# lumopt base_script: builds static parts of the simulation
# ─────────────────────────────────────────────────────────────────────────────

def make_base_script(cfg: SimulationConfig, n_free: int):
    """Return a callable that builds the static portion of the FDTD simulation.

    lumopt invokes this on a clean Lumerical session each iteration. After
    `_callback(fdtd)` returns, lumopt adds the FunctionDefinedPolygon (the
    freed-region SiN core) on top of the already-placed background.

    This builds: FDTD region, mesh override, materials, ports/source,
    transmission monitor, and all STATIC core rectangles (outer teeth,
    access waveguides). The freed-region (n_free innermost teeth on each
    side + cavity + d=n_free+1 narrow-shifted right tooth) is OMITTED
    from this script — lumopt's polygon fills it in.
    """
    from bragg_device import PiShiftBraggFDTD  # noqa: F401  — keep import for static analysis
    from runners.inverse_design.geometry_builder import freed_region_x_bounds

    def _callback(fdtd):
        # Build a "skeleton" device: same domain & mesh & ports as a normal
        # run, plus the static outer rectangles. The freed-region rectangles
        # are skipped — lumopt's polygon will fill them in.
        kwargs = cfg.to_device_kwargs()
        # We pass through n_free_inner_teeth so that bragg_device's _add_bragg_core
        # will know which teeth are "freed" (we then SUPPRESS those rectangles).
        kwargs["n_free_inner_teeth"] = n_free
        kwargs["inner_dw_nm"] = None       # static envelope for outer teeth
        kwargs["inner_shift_nm"] = [0.0] * n_free   # zero shift in skeleton (geometry filled by polygon)
        sim = PiShiftBraggFDTD(**kwargs)
        sim.fdtd = fdtd  # reuse lumopt's session
        sim._setup_materials()
        sim._add_fdtd_region()
        sim._add_aligned_mesh_override()
        # Add only the static rectangles (outside the freed-region x-bounds).
        # Easiest approach: add ALL rectangles via the existing builder, then
        # delete those whose x-extent overlaps the freed region. Simpler still:
        # use a flag (see TODO) to skip the freed rectangles inside _add_bragg_core.
        # For first-cut clarity we delete after the fact.
        sim._add_bragg_core()
        x_lo, x_hi = freed_region_x_bounds(cfg, n_free)
        # Delete any core_seg rectangle whose center sits strictly inside the freed region.
        # Lumerical: select by name pattern, get bounding boxes, delete matches.
        fdtd.eval(
            f"selectall; sel = numelements; for(i=1:sel){{select(get('name'));}}"
        )
        # Robust per-rectangle delete: iterate over our known prefixes for d ≤ n_free.
        for d in range(1, n_free + 1):
            for prefix in (f"L_narrow_{d}", f"L_wide_{d}", f"R_narrow_{d}", f"R_wide_{d}"):
                _delete_named_starting_with(fdtd, prefix)
        # Also delete cavity (re-emitted by polygon).
        _delete_named_starting_with(fdtd, "cavity")
        # And d=n_free+1 right narrow (its length depends on shift_N → polygon-controlled).
        # NOTE: wide_{n_free+1}_R width is static (envelope), but its X position depends
        # on shifts; simpler to let the polygon emit it too.
        _delete_named_starting_with(fdtd, f"R_narrow_{n_free + 1}")
        _delete_named_starting_with(fdtd, f"R_wide_{n_free + 1}")
        # Source/monitors are the same as a normal run.
        sim._add_source_and_monitors()
        # Update wavelength scan to the spec's narrowed band (10 nm default).
        # Pulled from cfg.spectral.
        sim.update_scan(
            cfg.spectral.center_wavelength_m,
            cfg.spectral.scan_width_nm,
            cfg.spectral.n_wl_points,
        )

    return _callback


def _delete_named_starting_with(fdtd, prefix: str) -> None:
    """Delete all Lumerical objects whose name starts with `prefix`."""
    fdtd.eval(
        f'''
        selectpartial("{prefix}");
        if (numselected() > 0) {{ delete; }}
        '''
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run one driver
# ─────────────────────────────────────────────────────────────────────────────

def run_inverse_design(
    cfg: SimulationConfig,
    spec: InverseDesignSpec,
    start_idx: int,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Run a single multi-start driver of the lumopt adjoint optimization."""
    from lumopt.optimization import Optimization                 # type: ignore
    from lumopt.figures_of_merit.modematch import ModeMatch       # type: ignore
    from lumopt.optimizers.scipy_optimizers import ScipyOptimizers  # type: ignore
    from runners.inverse_design.geometry_builder import make_lumopt_geometry
    from sim_helpers import peak_t_diagnostic

    spec.validate()
    starts = spec.get_starts()
    if not (0 <= start_idx < len(starts)):
        raise IndexError(f"start_idx {start_idx} out of range (have {len(starts)} starts).")
    p0 = starts[start_idx]

    # Configure cfg for the inverse-design path.
    cfg = copy.deepcopy(cfg)
    cfg.grating.n_free_inner_teeth = spec.n_free_inner_teeth
    cfg.grating.lengthen_cavity = spec.lengthen_cavity
    # Initial values seed the bragg_device path; lumopt's polygon callback
    # overrides during the actual optimization but bragg_device sees the
    # initial-point values when measuring baseline λ_resonance.
    init_kwargs = params_to_kwargs(p0, spec.n_free_inner_teeth)
    cfg.grating.inner_dw_nm = list(init_kwargs["inner_dw_nm"])
    cfg.grating.inner_shift_nm = list(init_kwargs["inner_shift_nm"])
    cfg.grating.cavity_width_m = init_kwargs["cavity_width_m"]

    # Output directory
    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "inverse_design", spec.label or "study")
    out_dir = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline λ_resonance
    if baseline_lambda_m is None:
        baseline_lambda_m = measure_baseline_lambda_m(cfg)

    # FOM: single-wavelength ModeMatch on Port_2.
    # target_T_fwd=lambda wl: 1.0 → maximize transmission at the resonance wavelength.
    fom = ModeMatch(
        monitor_name="Port_2",
        mode_number=1,
        direction="Forward",
        target_T_fwd=lambda wl: np.ones_like(wl),
        norm_p=1,
        target_fom=0.0,
    )

    # Geometry callback
    geometry = make_lumopt_geometry(cfg, spec, p0)

    # Optimizer
    optimizer = ScipyOptimizers(
        max_iter=spec.max_iter,
        method=spec.optimizer_method,
        scaling_factor=1.0,
        pgtol=spec.optimizer_pgtol,
        ftol=spec.optimizer_ftol,
    )

    # Optimization object
    base_script = make_base_script(cfg, spec.n_free_inner_teeth)
    opt = Optimization(
        base_script=base_script,
        wavelengths=np.array([baseline_lambda_m]),
        fom=fom,
        geometry=geometry,
        optimizer=optimizer,
        use_var_fdtd=False,
        hide_fdtd_cad=True,
        use_deps=True,
        store_all_simulations=True,
        save_global_index=False,
        label=f"{spec.label}_start{start_idx}",
        source_name="source",
    )
    if spec.use_concurrent_adjoint_solves:
        opt.continuation_max_iter = spec.max_iter

    print(f"[inverse_design] running lumopt for start{start_idx}, p0={p0} ...")
    fom_final, params_final = opt.run(working_dir=out_dir)

    # Post-optimization: re-simulate optimum and run peak_t_diagnostic.
    print(f"[inverse_design] post-opt verification at converged params ...")
    cfg_final = copy.deepcopy(cfg)
    fk = params_to_kwargs(list(params_final), spec.n_free_inner_teeth)
    cfg_final.grating.inner_dw_nm = list(fk["inner_dw_nm"])
    cfg_final.grating.inner_shift_nm = list(fk["inner_shift_nm"])
    cfg_final.grating.cavity_width_m = fk["cavity_width_m"]
    from runners.single.run_simulation import run_single_sim
    final_result = run_single_sim(cfg_final)
    wl = np.asarray(final_result.get("wl", final_result.get("wavelength")))
    T = np.asarray(final_result.get("T", final_result.get("transmission")))
    lam_peak, T_peak = peak_t_diagnostic(wl, T)

    summary = {
        "start_idx": int(start_idx),
        "label": spec.label,
        "p_initial": list(map(float, p0)),
        "p_final": list(map(float, params_final)),
        "fom_final": float(fom_final),
        "baseline_lambda_m": float(baseline_lambda_m),
        "true_peak_lambda_m": float(lam_peak),
        "true_peak_T": float(T_peak),
        "n_free_inner_teeth": spec.n_free_inner_teeth,
    }
    with open(os.path.join(out_dir, "final_params.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[inverse_design] start{start_idx} done. true peak T = {T_peak:.4f} at λ={lam_peak*1e9:.3f} nm")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load_spec(spec_module: str) -> tuple[InverseDesignSpec, SimulationConfig]:
    """Import a study module that defines `SPEC: InverseDesignSpec` and
    optionally `BASE: SimulationConfig`. Returns (spec, base_cfg)."""
    mod = importlib.import_module(spec_module)
    if not hasattr(mod, "SPEC"):
        raise AttributeError(f"{spec_module} must define a top-level SPEC: InverseDesignSpec.")
    spec = mod.SPEC
    base = getattr(mod, "BASE", None) or SimulationConfig()
    return spec, base


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="lumopt adjoint inverse design driver")
    ap.add_argument("--spec", required=True, help="Module with SPEC (e.g. runners.inverse_design.peak_t)")
    ap.add_argument("--start", type=int, required=True, help="Starting-point index (0..n_starts-1)")
    ap.add_argument("--baseline-lambda-nm", type=float, default=None,
                    help="Pre-measured baseline λ_resonance in nm (skips the baseline run)")
    ap.add_argument("--output-root", default=None,
                    help="Override output directory root")
    args = ap.parse_args(argv)

    spec, base = _load_spec(args.spec)
    base_lambda_m = args.baseline_lambda_nm * 1e-9 if args.baseline_lambda_nm else None
    run_inverse_design(
        cfg=base,
        spec=spec,
        start_idx=args.start,
        output_root=args.output_root,
        baseline_lambda_m=base_lambda_m,
    )


if __name__ == "__main__":
    main()
