"""
Lumerical-native optimization driver for the pi-shift Bragg grating.

This is a parallel implementation to runners/gradient_free_design/ that uses
Lumerical's BUILT-IN ``addsweep("Optimization")`` (Particle Swarm or Generic
Algorithm) instead of orchestrating PSO from Python. Same parametric structure
group (freed_group) with 5 user properties, same FOM (peak T over the bandgap),
same output schema as ``GradientFreeDesignSpec`` so downstream analysis tooling
needs no changes.

Why a separate module? The Python-driven PSO in
``runners.gradient_free_design.gradient_free_design`` is fully working and we
keep it untouched. This module is the result of debugging the previously-failed
``addsweep`` path; if it pans out it becomes the long-term default since it
hands FDTD dispatch (and concurrent particle evaluation) to Lumerical.

Knowledge-base references used while wiring this up:
  - "Optimization utility"                     /articles/360034922953
  - "Creating optimization tasks using a script" /articles/360034922973
  - "Y-branch optimization using particle swarm"  /articles/360042800333

Likely root causes of the previous silent ``runsweep`` no-op (addressed below):
  1. The FOM result must be a *fully qualified* ``getresult`` path that exists
     AFTER a fresh run. Pointing addsweepresult at a port-monitor T array does
     not work — addsweepresult needs an ANALYSIS GROUP that computes a SCALAR
     and exposes it via the analysis tree. Solved here by adding an analysis
     group named ``optimization_fom_script`` whose script does
     ``T_array = abs(getresult("FDTD::ports::Port_2","T").T);
     peak_T = max(T_array); ?peak_T;``  — the trailing ``?peak_T;`` exports
     ``peak_T`` as a result of the analysis group.
  2. Sweep-parameter ``Name`` must be a model path to a structure-group user
     property: ``::model::freed_group::dw_inner_1_nm``, ``Type=Length`` (units
     in meters when Type=Length).
  3. ``setsweep`` calls must come BEFORE ``addsweepparameter`` /
     ``addsweepresult`` and the optimizer ``type`` must read literally
     ``"Particle Swarm"`` (not ``"PSO"`` or ``"particle_swarm"``).
  4. ``Run mode = "Concurrent"`` and ``Concurrent simulations`` are set
     explicitly; otherwise the sweep can hang on a default we don't control.
  5. We do NOT call ``clearresources``/``addresource`` inside the sweep flow —
     athena_run_one.py already configures the FDTD GPU resource at FDTD-init
     time. Re-touching resources mid-sweep can confuse Lumerical's dispatcher.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import config as _cfg
from simulation_config import SimulationConfig

# Reuse the inverse_design + gradient_free_design helpers verbatim — DO NOT
# duplicate the geometry logic.
from runners.inverse_design.inverse_design import (
    params_to_kwargs,
    measure_baseline,
)
from runners.gradient_free_design.gradient_free_design import (
    _build_parametric_fsp,
    _set_freed_group_params,
    _make_fom_analysis_script,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spec dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LumericalNativeSpec:
    """Configuration for one Lumerical-native (addsweep('Optimization')) study.

    Field set mirrors GradientFreeDesignSpec so studies can be ported by
    swapping the dataclass name and changing ``algorithm`` if desired.
    """

    # ── Free parameters and bounds ──────────────────────────────────────────
    n_free_inner_teeth: int = 2
    dw_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(60.0, 400.0), (60.0, 400.0)]
    )
    shift_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 200.0), (0.0, 200.0)]
    )
    cavity_width_bounds_nm: Tuple[float, float] = (500.0, 1100.0)

    # ── Initial point (used as freed_group default + baseline measurement) ──
    initial_points: Optional[List[List[float]]] = None
    n_starts: int = 1
    seed: int = 0

    # ── Optimizer (Lumerical's built-in PSO / Generic Algorithm) ───────────
    # Exactly one of the two strings the Lumerical Optimization sweep accepts.
    algorithm: str = "Particle Swarm"
    population_size: int = 20            # particles per generation
    max_generations: int = 10            # ⇒ ~population × generations evals
    tolerance: float = 1e-4              # PSO/GA convergence criterion

    # Concurrent FDTD evaluations within ONE Lumerical Optimization run.
    # On Athena: --gpus=N for the SLURM job → up to N in flight at once.
    n_concurrent: int = 1

    # ── FOM band ───────────────────────────────────────────────────────────
    fom_window_nm: float = 10.0
    fom_n_points: int = 601

    # ── Mesh override on the freed region ──────────────────────────────────
    mesh_override_dxyz_nm: float = 10.0

    # ── Geometry constraints (mirror gradient_free_design) ─────────────────
    enforce_mirror_symmetry: bool = True
    lengthen_cavity: bool = True

    # ── Study metadata ─────────────────────────────────────────────────────
    label: str = ""

    # ─────────────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return 2 * self.n_free_inner_teeth + 1

    def all_bounds_nm(self) -> List[Tuple[float, float]]:
        return list(self.dw_bounds_nm) + list(self.shift_bounds_nm) + [self.cavity_width_bounds_nm]

    def validate(self) -> None:
        n = self.n_free_inner_teeth
        if n < 1:
            raise ValueError(f"n_free_inner_teeth must be >= 1, got {n}.")
        if n != 2:
            # _build_parametric_fsp's structure-group .lsf is unrolled for n=2.
            raise NotImplementedError(
                f"LumericalNativeSpec currently only supports n_free_inner_teeth=2 "
                f"(matches gradient_free_design's structure group), got {n}."
            )
        if len(self.dw_bounds_nm) != n:
            raise ValueError(
                f"dw_bounds_nm length ({len(self.dw_bounds_nm)}) "
                f"must equal n_free_inner_teeth ({n})."
            )
        if len(self.shift_bounds_nm) != n:
            raise ValueError(
                f"shift_bounds_nm length ({len(self.shift_bounds_nm)}) "
                f"must equal n_free_inner_teeth ({n})."
            )
        if self.algorithm not in ("Particle Swarm", "Generic Algorithm"):
            raise ValueError(
                f"algorithm must be 'Particle Swarm' or 'Generic Algorithm', "
                f"got {self.algorithm!r}."
            )

    def get_starts(self) -> List[List[float]]:
        self.validate()
        if self.initial_points is not None:
            return [list(p) for p in self.initial_points]
        return [[300.0, 300.0, 0.0, 0.0, 800.0]] * max(1, self.n_starts)

    def describe(self) -> str:
        return (f"LumericalNativeSpec(label={self.label!r})\n"
                f"  algorithm        = {self.algorithm}  (Lumerical built-in)\n"
                f"  pop x gens       = {self.population_size} x {self.max_generations} "
                f"= {self.population_size * self.max_generations} evals\n"
                f"  n_concurrent     = {self.n_concurrent}\n"
                f"  n_free_inner_teeth = {self.n_free_inner_teeth}\n"
                f"  dw_bounds_nm     = {self.dw_bounds_nm}\n"
                f"  shift_bounds_nm  = {self.shift_bounds_nm}\n"
                f"  cavity_bounds_nm = {self.cavity_width_bounds_nm}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. addsweep("Optimization") configuration
# ─────────────────────────────────────────────────────────────────────────────

# Sweep + analysis-group names. Kept as module-level constants so the analysis
# script and the addsweepresult call use the SAME name (a known footgun).
_SWEEP_NAME = "opt_peak_T"
_ANALYSIS_GROUP_NAME = "optimization_fom_script"
_FOM_SCALAR_NAME = "peak_T"


def _build_optimization_fom_analysis_group(fdtd) -> None:
    """Create the analysis group whose result feeds the Optimization sweep.

    The structure group rebuilt by _build_parametric_fsp adds a scratch
    'fom_extractor' analysis group, but we deliberately ADD A SECOND group
    here named ``optimization_fom_script``. Reasons:

      * The optimization sweep's result path must be stable and well-known
        ("::model::optimization_fom_script::peak_T"). Rebuilding it ourselves
        guarantees the script content matches what the sweep reads.
      * It keeps the gradient_free_design module's helper unchanged: this
        module's analysis-group plumbing is fully self-contained.

    The analysis script produces a single scalar named ``peak_T``. The trailing
    ``?peak_T;`` is what makes it appear in the analysis tree as a result
    available to ``addsweepresult``.
    """
    fdtd.addanalysisgroup()
    fdtd.set("name", _ANALYSIS_GROUP_NAME)
    script = _make_fom_analysis_script(transmission_monitor_path="FDTD::ports::Port_2")
    # _make_fom_analysis_script already ends with `?peak_T;` so peak_T appears
    # as a result of the analysis group after the sweep runs each FDTD job.
    fdtd.set("analysis script", script)


def _configure_optimization_sweep(fdtd, spec: LumericalNativeSpec, p0_nm: List[float]) -> None:
    """Build the addsweep('Optimization') node with PSO/GA, params, and FOM.

    This is the heart of the previously-failing wiring. Order matters:
      1) addsweep + setsweep('type', 'Optimization') registers the sweep node.
      2) Set optimizer 'type' (Particle Swarm / Generic Algorithm) BEFORE
         attaching parameters/results.
      3) Set 'maximize'=1, 'tolerance', 'maximum generations', 'generation size'.
      4) Set 'Run mode'='Concurrent' and 'Concurrent simulations'=N.
      5) Add each parameter as a sweep parameter struct (with model path,
         Type=Length, Min/Max in meters since Type=Length, plus a Start).
      6) Add the FOM as a sweep RESULT pointing at the analysis-group scalar.

    The Min/Max units are critical: when Type='Length', Lumerical interprets
    Min/Max as METERS, not nm. The user properties on freed_group store nm but
    the sweep parameter Type is 'Length' so Lumerical converts internally —
    we therefore pass Min/Max in METERS.
    """
    bounds_nm = spec.all_bounds_nm()
    p0_nm = list(p0_nm)

    # 1. Register the sweep node.
    fdtd.addsweep()
    fdtd.setsweep("sweep", "name", _SWEEP_NAME)
    # NB: 'type' here is the sweep KIND (Ranges / Values / Optimization / GA).
    fdtd.setsweep(_SWEEP_NAME, "type", "Optimization")

    # 2. Optimizer-level options. Naming follows the Lumerical KB
    #    "Creating optimization tasks using a script" (article 360034922973).
    fdtd.setsweep(_SWEEP_NAME, "optimizer type", spec.algorithm)
    fdtd.setsweep(_SWEEP_NAME, "maximize", 1)                 # 1 = maximize FOM
    fdtd.setsweep(_SWEEP_NAME, "tolerance", float(spec.tolerance))
    fdtd.setsweep(_SWEEP_NAME, "maximum generations", int(spec.max_generations))
    fdtd.setsweep(_SWEEP_NAME, "generation size", int(spec.population_size))

    # 3. Concurrent dispatch — explicit so the sweep doesn't fall back to a
    #    default that no-ops on headless lumapi sessions.
    try:
        fdtd.setsweep(_SWEEP_NAME, "Run mode", "Concurrent")
        fdtd.setsweep(_SWEEP_NAME, "Concurrent simulations", int(spec.n_concurrent))
    except Exception as exc:
        # Older Lumerical builds may not expose these; fall back silently.
        print(f"[lumerical_native] WARN: could not set concurrency on sweep "
              f"({exc}); Lumerical default applies.")

    # 4. Parameters (5 freed_group user properties).
    param_specs = [
        ("dw_inner_1_nm",    bounds_nm[0][0], bounds_nm[0][1], p0_nm[0]),
        ("dw_inner_2_nm",    bounds_nm[1][0], bounds_nm[1][1], p0_nm[1]),
        ("shift_inner_1_nm", bounds_nm[2][0], bounds_nm[2][1], p0_nm[2]),
        ("shift_inner_2_nm", bounds_nm[3][0], bounds_nm[3][1], p0_nm[3]),
        ("cavity_width_nm",  bounds_nm[4][0], bounds_nm[4][1], p0_nm[4]),
    ]
    for prop_name, lo_nm, hi_nm, start_nm in param_specs:
        # The 5th positional arg of addsweepparameter is a struct describing
        # the parameter. 'Name' must be a fully qualified model path to the
        # user property; 'Type' = 'Length' tells Lumerical to interpret
        # Min/Max/Start as meters.
        param_struct = {
            "Name":  f"::model::freed_group::{prop_name}",
            "Type":  "Length",
            "Min":   float(lo_nm)    * 1e-9,
            "Max":   float(hi_nm)    * 1e-9,
            "Start": float(start_nm) * 1e-9,
        }
        fdtd.addsweepparameter(_SWEEP_NAME, param_struct)

    # 5. FOM result. The Result path must resolve in the analysis tree AFTER a
    #    fresh run. The analysis group's `?peak_T;` makes it visible.
    result_struct = {
        "Name":   _FOM_SCALAR_NAME,
        "Result": f"::model::{_ANALYSIS_GROUP_NAME}::{_FOM_SCALAR_NAME}",
    }
    fdtd.addsweepresult(_SWEEP_NAME, result_struct)


def _read_optimization_result(fdtd) -> Tuple[List[float], float, List[dict]]:
    """Pull the optimum out of the completed Optimization sweep.

    Returns (best_params_nm, best_fom, history). The 'history' list mirrors
    the gradient_free_design.pso_history format:
        [{"gen": int, "best_fom": float, "best_params_nm": [..],
          "mean_fom": float}, ...]

    Lumerical's getsweepresult API for an Optimization sweep returns a struct
    with at least: ``best`` (the FOM at the optimum), the parameter values at
    the optimum (via getsweepresult on each parameter Name), and per-generation
    bookkeeping. The exact field set varies between Lumerical versions, so we
    defensively probe several known paths and fall back gracefully.
    """
    best_fom = float("nan")
    best_params: List[float] = [float("nan")] * 5
    history: List[dict] = []

    # Best FOM. The optimizer's result struct exposes the optimal FOM value
    # under either "fom" or the result Name (peak_T) depending on Lumerical
    # version.
    try:
        fom_struct = fdtd.getsweepresult(_SWEEP_NAME, _FOM_SCALAR_NAME)
        # fom_struct may be a numeric scalar OR a struct holding the optimum.
        if isinstance(fom_struct, dict):
            for k in ("best", "value", _FOM_SCALAR_NAME, "fom"):
                if k in fom_struct:
                    best_fom = float(np.asarray(fom_struct[k]).flatten()[0])
                    break
        else:
            arr = np.asarray(fom_struct).flatten()
            best_fom = float(arr[-1])  # the last entry is typically the optimum
    except Exception as exc:
        print(f"[lumerical_native] WARN: could not read FOM result: {exc}")

    # Best-parameter values. Each parameter Name has a getsweepresult entry.
    param_names = [
        "dw_inner_1_nm",
        "dw_inner_2_nm",
        "shift_inner_1_nm",
        "shift_inner_2_nm",
        "cavity_width_nm",
    ]
    for i, pname in enumerate(param_names):
        try:
            val = fdtd.getsweepresult(_SWEEP_NAME, pname)
            if isinstance(val, dict):
                # Newer Lumerical wraps optimum in {'best': ..., ...}.
                for k in ("best", "value", pname):
                    if k in val:
                        v_m = float(np.asarray(val[k]).flatten()[0])
                        break
                else:
                    v_m = float("nan")
            else:
                arr = np.asarray(val).flatten()
                v_m = float(arr[-1])
            # Lumerical stores Length-type params in METERS; convert to nm
            # to match GradientFreeDesignSpec output schema.
            best_params[i] = v_m * 1e9
        except Exception as exc:
            print(f"[lumerical_native] WARN: could not read param {pname!r}: {exc}")

    # Per-generation history. Optional — Lumerical exposes a result called
    # something like 'fom_history' in some versions. We try and fall back to
    # an empty history if absent (the final-best-only summary is still useful).
    for hist_key in ("fom_history", "history", "best_fom_history"):
        try:
            h = fdtd.getsweepresult(_SWEEP_NAME, hist_key)
            arr = np.asarray(h).flatten()
            for g, fom in enumerate(arr):
                history.append({
                    "gen": int(g),
                    "best_fom": float(fom),
                    "best_params_nm": list(map(float, best_params)),
                    "mean_fom": float("nan"),
                })
            break
        except Exception:
            continue

    return list(map(float, best_params)), float(best_fom), history


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main driver
# ─────────────────────────────────────────────────────────────────────────────


def run_lumerical_native(
    cfg: SimulationConfig,
    spec: LumericalNativeSpec,
    start_idx: int = 0,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Run one Lumerical-native (addsweep('Optimization')) study."""
    import sys as _sys
    _lum_dir = os.path.dirname(_cfg.LUMAPI_PATH)
    if _lum_dir and _lum_dir not in _sys.path:
        _sys.path.insert(0, _lum_dir)
    import lumapi  # type: ignore
    from runners.single.run_simulation import run_single_sim

    spec.validate()
    starts = spec.get_starts()
    if not (0 <= start_idx < len(starts)):
        raise IndexError(f"start_idx {start_idx} out of range (have {len(starts)} starts).")
    p0 = list(starts[start_idx])

    # Configure cfg for the inverse-design path (same as gradient_free_design).
    cfg = copy.deepcopy(cfg)
    cfg.grating.n_free_inner_teeth = spec.n_free_inner_teeth
    cfg.grating.lengthen_cavity = spec.lengthen_cavity
    init_kwargs = params_to_kwargs(p0, spec.n_free_inner_teeth)
    cfg.grating.inner_dw_nm = list(init_kwargs["inner_dw_nm"])
    cfg.grating.inner_shift_nm = list(init_kwargs["inner_shift_nm"])
    cfg.grating.cavity_width_m = init_kwargs["cavity_width_m"]

    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "lumerical_native_optimization",
                                   spec.label or "study")
    out_dir = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline λ_resonance + peak T at the starting geometry (skipped if the
    # caller already measured it).
    initial_peak_T: Optional[float] = None
    if baseline_lambda_m is None:
        baseline_lambda_m, initial_peak_T = measure_baseline(cfg)

    print(f"[lumerical_native] building parametric .fsp ...")
    fdtd = lumapi.FDTD(hide=True)
    fsp_path = os.path.join(out_dir, "lumerical_native.fsp")
    p_final: List[float] = list(p0)
    fom_final = float("nan")
    history: List[dict] = []
    try:
        # Build the static skeleton + parametric freed_group + monitors via
        # the gradient_free_design helper. This also leaves the FDTD in
        # layout mode with the geometry rendered (runsetup() is called there).
        _build_parametric_fsp(fdtd, cfg, spec, p0, baseline_lambda_m)

        # Add OUR analysis group on top — separate name from the helper's
        # 'fom_extractor' so the sweep result path is stable and unambiguous.
        _build_optimization_fom_analysis_group(fdtd)

        # Start the freed_group at the user's chosen p0 (helper already did
        # this, but we re-set defensively in case the helper's adduserprop
        # defaults drifted).
        _set_freed_group_params(fdtd, p0)

        # Configure the addsweep('Optimization') node.
        _configure_optimization_sweep(fdtd, spec, p0)

        # Save the .fsp BEFORE running the sweep so we can debug it later.
        fdtd.save(fsp_path)
        print(f"[lumerical_native] saved pre-sweep .fsp: {fsp_path}")

        print(f"[lumerical_native] running Lumerical Optimization sweep "
              f"(algorithm={spec.algorithm}, pop={spec.population_size}, "
              f"gens={spec.max_generations}, n_concurrent={spec.n_concurrent}) ...")
        fdtd.runsweep(_SWEEP_NAME)

        # Save again after the sweep so the .fsp contains the result tree.
        fdtd.save(fsp_path)

        p_final, fom_final, history = _read_optimization_result(fdtd)
        # If the result-read path returned NaNs (Lumerical version mismatch on
        # result keys), fall back to the freed_group's current user-property
        # values. The sweep updates them to the optimum on completion.
        if not all(np.isfinite(p_final)):
            print("[lumerical_native] WARN: result-tree read returned NaN params; "
                  "falling back to freed_group user-property values.")
            try:
                p_final = [
                    float(fdtd.getnamed("freed_group", "dw_inner_1_nm")),
                    float(fdtd.getnamed("freed_group", "dw_inner_2_nm")),
                    float(fdtd.getnamed("freed_group", "shift_inner_1_nm")),
                    float(fdtd.getnamed("freed_group", "shift_inner_2_nm")),
                    float(fdtd.getnamed("freed_group", "cavity_width_nm")),
                ]
            except Exception as exc:
                print(f"[lumerical_native] could not read freed_group fallback: {exc}")
                p_final = list(p0)
    finally:
        fdtd.close()

    # Post-optimization verification at higher mesh accuracy (mirrors
    # gradient_free_design.run_gradient_free_design).
    print("[lumerical_native] post-opt verification at converged params (accurate mesh) ...")
    cfg_final = copy.deepcopy(cfg)
    cfg_final.mesh.simulation_mode = "accurate"
    fk = params_to_kwargs(list(p_final), spec.n_free_inner_teeth)
    cfg_final.grating.inner_dw_nm = list(fk["inner_dw_nm"])
    cfg_final.grating.inner_shift_nm = list(fk["inner_shift_nm"])
    cfg_final.grating.cavity_width_m = fk["cavity_width_m"]
    final_result = run_single_sim(cfg_final)
    lam_peak = float(final_result["resonance_wavelength_nm"]) * 1e-9
    T_peak   = float(final_result["resonance_transmission"])

    summary = {
        "start_idx":          int(start_idx),
        "label":              spec.label,
        "p_initial":          list(map(float, p0)),
        "p_final":            list(map(float, p_final)),
        "fom_final":          float(fom_final),
        "baseline_lambda_m":  float(baseline_lambda_m),
        "initial_peak_T":     (None if initial_peak_T is None else float(initial_peak_T)),
        "true_peak_lambda_m": float(lam_peak),
        "true_peak_T":        float(T_peak),
        "delta_peak_T":       (None if initial_peak_T is None else float(T_peak - initial_peak_T)),
        "n_free_inner_teeth": spec.n_free_inner_teeth,
        "algorithm":          f"{spec.algorithm} (Lumerical native addsweep)",
        "population_size":    spec.population_size,
        "max_generations":    spec.max_generations,
        "n_concurrent":       spec.n_concurrent,
        "pso_history":        history,
    }
    with open(os.path.join(out_dir, "final_params.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    if initial_peak_T is not None:
        print(f"[lumerical_native] done. peak T: initial = {initial_peak_T:.4f}  "
              f"-> final = {T_peak:.4f}  (delta = {T_peak - initial_peak_T:+.4f}) "
              f"at lambda = {lam_peak * 1e9:.3f} nm")
    else:
        print(f"[lumerical_native] done. true peak T = {T_peak:.4f} "
              f"at lambda = {lam_peak * 1e9:.3f} nm")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────


def _load_spec(spec_module: str) -> Tuple[LumericalNativeSpec, SimulationConfig]:
    mod = importlib.import_module(spec_module)
    if not hasattr(mod, "SPEC"):
        raise AttributeError(
            f"{spec_module} must define a top-level SPEC: LumericalNativeSpec."
        )
    return mod.SPEC, getattr(mod, "BASE", None) or SimulationConfig()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Lumerical-native addsweep('Optimization') runner")
    ap.add_argument("--spec", required=True,
                    help="Module with SPEC "
                         "(e.g. runners.lumerical_native_optimization.optimize_transmission)")
    ap.add_argument("--start", type=int, default=0,
                    help="Starting-point index (default 0)")
    ap.add_argument("--baseline-lambda-nm", type=float, default=None,
                    help="Pre-measured baseline lambda_resonance in nm "
                         "(skips the baseline run)")
    ap.add_argument("--output-root", default=None, help="Override output directory root")
    args = ap.parse_args(argv)

    spec, base = _load_spec(args.spec)
    base_lambda_m = args.baseline_lambda_nm * 1e-9 if args.baseline_lambda_nm else None
    run_lumerical_native(
        cfg=base,
        spec=spec,
        start_idx=args.start,
        output_root=args.output_root,
        baseline_lambda_m=base_lambda_m,
    )


if __name__ == "__main__":
    main()
