"""
Lumerical-native gradient-free inverse design for the pi-shift Bragg grating.

Drives `addsweep("Optimization")` (Particle Swarm or Generic Algorithm) from
inside an lumapi.FDTD session. The freed region is rendered as a parametric
structure group whose construction script reads 5 user properties and rebuilds
11 named rectangles. The optimization sweep targets those user properties; the
FOM is `max(T(λ))` over the bandgap window — a non-differentiable quantity, fine
for gradient-free.

Layout of the parameter vector p (length 2*N+1, same convention as Option A):
    p = [dw_1, ..., dw_N,           # full corrugation depth (W_wide-W_narrow), nm
         shift_1, ..., shift_N,      # gap shift, nm
         cavity_width_nm]
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

# We share the geometry math + static-skeleton + base helpers with Option A.
from runners.inverse_design.inverse_design import (
    freed_region_x_bounds,
    freed_region_named_rects,
    params_to_kwargs,
    regular_grating_start,
    measure_baseline,
    _build_static_skeleton,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spec dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GradientFreeDesignSpec:
    """Configuration for one gradient-free inverse-design study."""

    # ── Free parameters and bounds ──────────────────────────────────────────
    n_free_inner_teeth: int = 2
    dw_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(60.0, 400.0), (60.0, 400.0)]
    )
    shift_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 200.0), (0.0, 200.0)]
    )
    cavity_width_bounds_nm: Tuple[float, float] = (500.0, 1100.0)

    # ── Initial point (PSO uses this as the centre for swarm seeding) ──────
    initial_points: Optional[List[List[float]]] = None
    n_starts: int = 1
    seed: int = 0

    # ── Optimizer (Lumerical's built-in PSO / Generic Algorithm) ───────────
    # algorithm: "Particle Swarm" or "Generic Algorithm".
    algorithm: str = "Particle Swarm"
    population_size: int = 20            # particles per generation
    max_generations: int = 10            # ⇒ ~200 evals total
    tolerance: float = 1e-4              # PSO convergence criterion

    # Concurrent FDTD evaluations within ONE Lumerical Optimization run.
    # On Athena: --gpus=4 for the SLURM job → up to 4 in flight at once.
    n_concurrent: int = 4

    # ── FOM band ───────────────────────────────────────────────────────────
    # The transmission monitor records T(λ) over [λ0 ± fom_window_nm/2].
    # FOM is max(T) over that band (non-differentiable, fine for PSO).
    fom_window_nm: float = 10.0
    fom_n_points: int = 601

    # ── Mesh override on the freed region ──────────────────────────────────
    mesh_override_dxyz_nm: float = 10.0

    # ── Geometry constraints (mirror Option A) ─────────────────────────────
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
        # PSO doesn't strictly need a starting point but we keep one for
        # API parity with Option A and for the baseline measurement.
        return [[300.0, 300.0, 0.0, 0.0, 800.0]] * max(1, self.n_starts)

    def describe(self) -> str:
        return (f"GradientFreeDesignSpec(label={self.label!r})\n"
                f"  algorithm        = {self.algorithm}\n"
                f"  pop × gens       = {self.population_size} × {self.max_generations} "
                f"= {self.population_size * self.max_generations} evals\n"
                f"  n_concurrent     = {self.n_concurrent}\n"
                f"  n_free_inner_teeth = {self.n_free_inner_teeth}\n"
                f"  dw_bounds_nm     = {self.dw_bounds_nm}\n"
                f"  shift_bounds_nm  = {self.shift_bounds_nm}\n"
                f"  cavity_bounds_nm = {self.cavity_width_bounds_nm}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lumerical-script generation: parametric structure-group construction.
#    The .lsf script reads the group's user properties and emits 11 named
#    rectangles for the freed region.
# ─────────────────────────────────────────────────────────────────────────────


def _make_freed_group_script(cfg: SimulationConfig, n_free: int, core_material_name: str) -> str:
    """Build the .lsf construction script for the parametric freed-region group.

    The script reads `dw_inner_X_nm`, `shift_inner_X_nm`, `cavity_width_nm`
    (group user properties) and emits 11 rectangles inside the group.
    """
    if n_free != 2:
        # For now we hardwire n_free=2 (the .lsf math below is unrolled).
        # Future: emit a loop in .lsf for arbitrary n_free.
        raise NotImplementedError(
            f"_make_freed_group_script: only n_free=2 is supported, got {n_free}.")

    pitch = cfg.grating.pitch_m
    half_p = pitch / 2.0
    avg_w = cfg.geometry.avg_corrugation_width_m
    core_h = cfg.geometry.core_height_m
    n_total = cfg.grating.n_periods_each_side
    full_dw = cfg.geometry.corrugation_depth_m
    if cfg.grating.cavity_neg_detuning_nm != 0.0:
        cav_len_baseline = half_p - cfg.grating.cavity_neg_detuning_nm * 1e-9
    else:
        cav_len_baseline = half_p
    x_grating_end = n_total * pitch + cav_len_baseline / 2.0

    # Lumerical .lsf does NOT have Python-style format strings, so we embed
    # the numeric constants by Python f-string substitution at .fsp build time.
    return f"""# Parametric freed-region geometry. Runs whenever a user property changes.
# User props: dw_inner_1_nm, dw_inner_2_nm, shift_inner_1_nm, shift_inner_2_nm,
#             cavity_width_nm.
deleteall;

dw1 = dw_inner_1_nm * 1e-9;
dw2 = dw_inner_2_nm * 1e-9;
sh1 = shift_inner_1_nm * 1e-9;
sh2 = shift_inner_2_nm * 1e-9;
cav_w = cavity_width_nm * 1e-9;

half_p = {half_p:.15g};
avg_w  = {avg_w:.15g};
core_h = {core_h:.15g};
full_dw = {full_dw:.15g};
x_grating_end = {x_grating_end:.15g};
n_total = {n_total};
n_free = {n_free};
cav_len_baseline = {cav_len_baseline:.15g};

cav_len = cav_len_baseline + 2*(sh1 + sh2);

Wn_1 = avg_w - 0.5*dw1;  Ww_1 = avg_w + 0.5*dw1;
Wn_2 = avg_w - 0.5*dw2;  Ww_2 = avg_w + 0.5*dw2;
Wn_3 = avg_w - 0.5*full_dw;  Ww_3 = avg_w + 0.5*full_dw;

x_freed_start = -x_grating_end + (n_total - n_free) * (2*half_p);
x = x_freed_start;

# === Left freed teeth: d=2 (segs 00,01), d=1 (segs 02,03) ===
xs = half_p - sh2;
addrect; set("name","freed_seg_00"); set("material","{core_material_name}");
set("x min",x); set("x max",x+xs); set("y",0); set("y span",Wn_2);
set("z",0); set("z span",core_h);
x = x + xs;

addrect; set("name","freed_seg_01"); set("material","{core_material_name}");
set("x min",x); set("x max",x+half_p); set("y",0); set("y span",Ww_2);
set("z",0); set("z span",core_h);
x = x + half_p;

xs = half_p - sh1;
addrect; set("name","freed_seg_02"); set("material","{core_material_name}");
set("x min",x); set("x max",x+xs); set("y",0); set("y span",Wn_1);
set("z",0); set("z span",core_h);
x = x + xs;

addrect; set("name","freed_seg_03"); set("material","{core_material_name}");
set("x min",x); set("x max",x+half_p); set("y",0); set("y span",Ww_1);
set("z",0); set("z span",core_h);
x = x + half_p;

# === Cavity (seg 04) ===
addrect; set("name","freed_seg_04"); set("material","{core_material_name}");
set("x min",x); set("x max",x+cav_len); set("y",0); set("y span",cav_w);
set("z",0); set("z span",core_h);
x = x + cav_len;

# === Right freed teeth: d=1 (segs 05,06), d=2 (segs 07,08), d=3 (segs 09,10) ===
# d=1 narrow: NOT shortened (innermost right side, no preceding shift)
addrect; set("name","freed_seg_05"); set("material","{core_material_name}");
set("x min",x); set("x max",x+half_p); set("y",0); set("y span",Wn_1);
set("z",0); set("z span",core_h);
x = x + half_p;

addrect; set("name","freed_seg_06"); set("material","{core_material_name}");
set("x min",x); set("x max",x+half_p); set("y",0); set("y span",Ww_1);
set("z",0); set("z span",core_h);
x = x + half_p;

# d=2 narrow shortened by sh1 (right side: shift_{{d-1}})
xs = half_p - sh1;
addrect; set("name","freed_seg_07"); set("material","{core_material_name}");
set("x min",x); set("x max",x+xs); set("y",0); set("y span",Wn_2);
set("z",0); set("z span",core_h);
x = x + xs;

addrect; set("name","freed_seg_08"); set("material","{core_material_name}");
set("x min",x); set("x max",x+half_p); set("y",0); set("y span",Ww_2);
set("z",0); set("z span",core_h);
x = x + half_p;

# d=3 narrow shortened by sh2
xs = half_p - sh2;
addrect; set("name","freed_seg_09"); set("material","{core_material_name}");
set("x min",x); set("x max",x+xs); set("y",0); set("y span",Wn_3);
set("z",0); set("z span",core_h);
x = x + xs;

addrect; set("name","freed_seg_10"); set("material","{core_material_name}");
set("x min",x); set("x max",x+half_p); set("y",0); set("y span",Ww_3);
set("z",0); set("z span",core_h);
"""


def _make_fom_analysis_script(transmission_monitor_path: str = "FDTD::ports::Port_2") -> str:
    """Lumerical analysis script that computes peak T over the recorded band.

    Sets a result `peak_T` accessible via `getsweepresult` paths after each
    FDTD run.

    TODO (future runs): port sim_helpers.find_bragg_resonance into .lsf so the
    FOM is the cavity-resonance peak (rejecting sidelobe ripple and band-edge
    artifacts) rather than max(T). Keep raw-max here for backward compatibility
    with histories from existing runs.
    """
    return f"""# fom_extractor analysis script: peak modal T from the transmission monitor.
T_struct = getresult("{transmission_monitor_path}", "T");
T_array = abs(T_struct.T);
peak_T = max(T_array);
# In Lumerical 2026R1, the trailing `?` operator just prints; it does NOT
# guarantee `peak_T` appears as a queryable analysis-group result for
# getsweepresult / addsweepresult. addresult() makes the result explicit.
addresult("peak_T", peak_T);
?peak_T;
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the parameterized .fsp via lumapi
# ─────────────────────────────────────────────────────────────────────────────


def _build_parametric_fsp(fdtd, cfg: SimulationConfig, spec: GradientFreeDesignSpec,
                          initial_p, target_lambda_m: float) -> None:
    """Construct the full .fsp inside `fdtd`: static skeleton + parametric
    freed group + opt-region monitor + finite-fine override mesh + analysis
    group computing peak T.
    """
    from bragg_device import PiShiftBraggFDTD

    n_free = spec.n_free_inner_teeth

    # Build the static skeleton via the same path Option A uses.
    kwargs = cfg.to_device_kwargs()
    kwargs["n_free_inner_teeth"] = n_free
    kwargs["inner_dw_nm"] = None
    kwargs["inner_shift_nm"] = [0.0] * n_free
    sim = PiShiftBraggFDTD(**kwargs)
    sim.fdtd = fdtd
    sim._setup_materials()
    sim._add_fdtd_region()
    sim._add_aligned_mesh_override()
    _build_static_skeleton(sim, cfg, n_free)
    sim._add_source_and_monitors()

    # Wide scan over the bandgap: PSO needs T(λ) full band so its FOM
    # max(T) sees the resonance regardless of where it sits.
    sim.update_scan(
        target_lambda_m,
        spec.fom_window_nm,
        spec.fom_n_points,
    )

    # Resolve core material name (set by _setup_materials on a static rect).
    try:
        core_material = str(fdtd.getnamed("wg_left_inf_1", "material"))
    except Exception:
        core_material = "Si3N4 (Silicon Nitride) - Luke"

    # Add fine override mesh over the freed region (skip if 0 — at
    # n_periods=80 the fine override makes FDTD intractable).
    x_lo, x_hi = freed_region_x_bounds(cfg, n_free)
    if spec.mesh_override_dxyz_nm and spec.mesh_override_dxyz_nm > 0:
        mesh_dx = spec.mesh_override_dxyz_nm * 1e-9
        margin = 2.0 * mesh_dx
        fdtd.addmesh()
        fdtd.set("name", "mesh_override_freed")
        fdtd.set("x min", x_lo - margin)
        fdtd.set("x max", x_hi + margin)
        fdtd.set("y", 0.0)
        fdtd.set("y span", cfg.geometry.width_wide_m * 1.4 + 2 * margin)
        fdtd.set("z", 0.0)
        fdtd.set("z span", cfg.geometry.core_height_m * 1.4 + 2 * margin)
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)
        fdtd.set("dx", mesh_dx)
        fdtd.set("dy", mesh_dx)
        fdtd.set("dz", mesh_dx)

    # Add the parametric structure group for the freed region.
    fdtd.addstructuregroup()
    fdtd.set("name", "freed_group")
    fdtd.set("x", 0.0)

    fdtd.adduserprop("dw_inner_1_nm",     0, float(initial_p[0]))
    fdtd.adduserprop("dw_inner_2_nm",     0, float(initial_p[1]))
    fdtd.adduserprop("shift_inner_1_nm",  0, float(initial_p[2]))
    fdtd.adduserprop("shift_inner_2_nm",  0, float(initial_p[3]))
    fdtd.adduserprop("cavity_width_nm",   0, float(initial_p[4]))

    group_script = _make_freed_group_script(cfg, n_free, core_material)
    fdtd.set("script", group_script)

    # Analysis group that computes peak T from the transmission port's modal T.
    fdtd.addanalysisgroup()
    fdtd.set("name", "fom_extractor")
    analysis_script = _make_fom_analysis_script(transmission_monitor_path="FDTD::ports::Port_2")
    fdtd.set("analysis script", analysis_script)

    # Trigger the structure group's construction script so the freed-region
    # rectangles are actually generated. Without this, the .fsp stores the
    # script but doesn't render rectangles until the GUI/sweep triggers it.
    fdtd.runsetup()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Python-driven Particle Swarm Optimization (PSO) over the freed-region
#    parameters. Each particle evaluation calls Lumerical FDTD (via lumapi)
#    on the parametrised structure group → reads peak T → returns scalar.
#
# We previously tried `addsweep("Optimization")`, but Lumerical's Optimization-
# sweep object's `runsweep` / `getsweepresult` API has subtle requirements
# that silently no-op'd in our setup. Driving PSO directly from Python is
# much simpler and gives us complete control over particle dispatch.
# Lumerical still does all the FDTD; we just orchestrate the search.
# ─────────────────────────────────────────────────────────────────────────────


def _set_freed_group_params(fdtd, params_nm) -> None:
    """Push a 5-vector of parameter values (in nm) into the freed_group user
    properties, then trigger the structure group's construction script."""
    fdtd.switchtolayout()
    fdtd.setnamed("freed_group", "dw_inner_1_nm",     float(params_nm[0]))
    fdtd.setnamed("freed_group", "dw_inner_2_nm",     float(params_nm[1]))
    fdtd.setnamed("freed_group", "shift_inner_1_nm",  float(params_nm[2]))
    fdtd.setnamed("freed_group", "shift_inner_2_nm",  float(params_nm[3]))
    fdtd.setnamed("freed_group", "cavity_width_nm",   float(params_nm[4]))
    # Trigger the construction script so the rectangles update.
    fdtd.runsetup()


def _evaluate_particle(fdtd, params_nm) -> float:
    """Update the geometry to params_nm, run FDTD, return cavity peak T.

    Preferred path: modal |S21|^2 from the port's mode expansion. Port "T"
    is flux-based (includes cladding radiation crossing the port plane) and
    overestimates cavity transmission, so we avoid it as primary FOM.

    Fallback path: if the modal-expansion result isn't available (Lumerical
    sometimes doesn't populate it for certain geometries), we still apply
    find_bragg_resonance to port "T" — sidelobe and band-edge artifacts
    are rejected, but cladding radiation may still contribute. We log the
    fallback so we can audit later.
    """
    from sim_helpers import find_bragg_resonance

    _set_freed_group_params(fdtd, params_nm)
    fdtd.run()

    try:
        # Primary: modal |S21|^2 (TE0-projected forward transmission).
        res = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
        S21 = np.squeeze(res["S"])
        wl  = np.squeeze(res["lambda"]).flatten()
        T = np.abs(np.asarray(S21).flatten()) ** 2
        source = "modal"
    except Exception as exc:
        # Fallback: port "T" is always available. find_bragg_resonance still
        # rejects out-of-band and sidelobe artifacts.
        print(f"[pso] modal expansion unavailable ({exc}); falling back to port T.")
        res = fdtd.getresult("FDTD::ports::Port_2", "T")
        T  = np.asarray(np.squeeze(res["T"])).flatten()
        wl_m = np.asarray(np.squeeze(res["lambda"])).flatten()
        wl = wl_m * 1e9 if wl_m.max() < 1e-3 else wl_m
        source = "flux"

    idx = find_bragg_resonance(wl, T)
    peak_T = float(T[idx])
    fdtd.switchtolayout()
    return peak_T


def _pso_optimize(fdtd, spec: GradientFreeDesignSpec, p0_nm,
                   incremental_save_path: Optional[str] = None) -> Tuple[List[float], float, List[dict]]:
    """Standard global-best PSO over the 5 freed-region parameters.

    Returns (best_params_nm, best_fom, history).

    If incremental_save_path is given, writes a partial JSON after each
    generation so walltime-cut runs still produce useful output.
    """
    rng = np.random.default_rng(spec.seed)
    bounds = np.asarray(spec.all_bounds_nm(), dtype=float)   # shape (5, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_dim = len(p0_nm)
    pop = int(spec.population_size)
    n_gens = int(spec.max_generations)

    # PSO hyperparameters (standard "canonical PSO" values).
    w = 0.729        # inertia
    c1 = 1.494       # cognitive (personal best pull)
    c2 = 1.494       # social (global best pull)
    v_max = 0.2 * (hi - lo)

    # Initialize positions — first particle at p0, rest random in bounds.
    X = rng.uniform(lo, hi, size=(pop, n_dim))
    X[0, :] = np.asarray(p0_nm, dtype=float)
    V = rng.uniform(-v_max, v_max, size=(pop, n_dim))

    # Evaluate initial swarm.
    F = np.empty(pop)
    for i in range(pop):
        F[i] = _evaluate_particle(fdtd, X[i])
        print(f"[pso] gen 0  particle {i+1}/{pop}: peak_T = {F[i]:.6f}  "
              f"params = {[f'{x:.2f}' for x in X[i]]}")

    P_best = X.copy()
    F_best = F.copy()
    g_idx = int(np.argmax(F_best))
    G_best = P_best[g_idx].copy()
    G_fom  = float(F_best[g_idx])

    history: List[dict] = [{
        "gen": 0,
        "best_fom": G_fom,
        "best_params_nm": list(map(float, G_best)),
        "mean_fom": float(np.mean(F)),
    }]
    print(f"[pso] gen 0 done. global best peak_T = {G_fom:.6f} "
          f"params = {[f'{x:.2f}' for x in G_best]}")
    if incremental_save_path:
        with open(incremental_save_path, "w") as fp:
            json.dump({"history": history,
                       "best_params_nm": list(map(float, G_best)),
                       "best_fom": G_fom,
                       "completed_generations": 0}, fp, indent=2)

    # Generations
    for g in range(1, n_gens + 1):
        r1 = rng.random((pop, n_dim))
        r2 = rng.random((pop, n_dim))
        V = (w * V
             + c1 * r1 * (P_best - X)
             + c2 * r2 * (G_best[None, :] - X))
        V = np.clip(V, -v_max, v_max)
        X = X + V
        # Reflect at bounds (don't clip — particles bounce off walls).
        below = X < lo
        above = X > hi
        X = np.where(below, 2 * lo - X, X)
        X = np.where(above, 2 * hi - X, X)
        X = np.clip(X, lo, hi)   # final safety clip

        for i in range(pop):
            F[i] = _evaluate_particle(fdtd, X[i])
            print(f"[pso] gen {g}  particle {i+1}/{pop}: peak_T = {F[i]:.6f}  "
                  f"params = {[f'{x:.2f}' for x in X[i]]}")
            if F[i] > F_best[i]:
                F_best[i] = F[i]
                P_best[i] = X[i].copy()

        g_idx = int(np.argmax(F_best))
        if F_best[g_idx] > G_fom + 1e-9:
            G_fom = float(F_best[g_idx])
            G_best = P_best[g_idx].copy()

        history.append({
            "gen": g,
            "best_fom": G_fom,
            "best_params_nm": list(map(float, G_best)),
            "mean_fom": float(np.mean(F)),
        })
        print(f"[pso] gen {g} done. global best peak_T = {G_fom:.6f} "
              f"params = {[f'{x:.2f}' for x in G_best]}")
        if incremental_save_path:
            with open(incremental_save_path, "w") as fp:
                json.dump({"history": history,
                           "best_params_nm": list(map(float, G_best)),
                           "best_fom": G_fom,
                           "completed_generations": g}, fp, indent=2)

        # Convergence check: stop if global best hasn't improved by tolerance
        # for a couple of generations.
        if g >= 2 and abs(history[-1]["best_fom"] - history[-3]["best_fom"]) < spec.tolerance:
            print(f"[pso] converged at generation {g} (tol={spec.tolerance}).")
            break

    return list(map(float, G_best)), float(G_fom), history


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main driver
# ─────────────────────────────────────────────────────────────────────────────


def run_gradient_free_design(
    cfg: SimulationConfig,
    spec: GradientFreeDesignSpec,
    start_idx: int = 0,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Run one Lumerical-native gradient-free optimization."""
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

    # Configure cfg for the inverse-design path.
    cfg = copy.deepcopy(cfg)
    cfg.grating.n_free_inner_teeth = spec.n_free_inner_teeth
    cfg.grating.lengthen_cavity = spec.lengthen_cavity
    init_kwargs = params_to_kwargs(p0, spec.n_free_inner_teeth)
    cfg.grating.inner_dw_nm = list(init_kwargs["inner_dw_nm"])
    cfg.grating.inner_shift_nm = list(init_kwargs["inner_shift_nm"])
    cfg.grating.cavity_width_m = init_kwargs["cavity_width_m"]

    # Output directory
    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "gradient_free_design",
                                   spec.label or "study")
    out_dir = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline λ_resonance + peak T at the starting geometry.
    initial_peak_T: Optional[float] = None
    if baseline_lambda_m is None:
        baseline_lambda_m, initial_peak_T = measure_baseline(cfg)

    print(f"[gradient_free_design] building parametric .fsp ...")
    fdtd = lumapi.FDTD(hide=True)
    fsp_path = os.path.join(out_dir, "gf_design.fsp")
    p_final: List[float] = list(p0)
    fom_final = float("nan")
    history: List[dict] = []
    try:
        _build_parametric_fsp(fdtd, cfg, spec, p0, baseline_lambda_m)

        fdtd.save(fsp_path)
        incremental_path = os.path.join(out_dir, "pso_incremental.json")
        print(f"[gradient_free_design] running PSO "
              f"(pop={spec.population_size}, gens={spec.max_generations}) ...")
        p_final, fom_final, history = _pso_optimize(fdtd, spec, p0,
                                                     incremental_save_path=incremental_path)
    finally:
        fdtd.close()

    # Post-optimization verification: re-simulate the converged geometry via
    # the full broadband baseline pipeline at HIGHER mesh accuracy so the
    # headline true_peak_T is physics-precise even though the PSO loop ran
    # at the coarser optimization mesh.
    print("[gradient_free_design] post-opt verification at converged params (accurate mesh) ...")
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
        "algorithm":          "Particle Swarm (Python-driven)",
        "population_size":    spec.population_size,
        "max_generations":    spec.max_generations,
        "n_concurrent":       spec.n_concurrent,
        "pso_history":        history,
    }
    with open(os.path.join(out_dir, "final_params.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    if initial_peak_T is not None:
        print(f"[gradient_free_design] done. peak T: initial = {initial_peak_T:.4f}  "
              f"→ final = {T_peak:.4f}  (Δ = {T_peak - initial_peak_T:+.4f}) "
              f"at λ = {lam_peak * 1e9:.3f} nm")
    else:
        print(f"[gradient_free_design] done. true peak T = {T_peak:.4f} "
              f"at λ = {lam_peak * 1e9:.3f} nm")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────


def _load_spec(spec_module: str) -> Tuple[GradientFreeDesignSpec, SimulationConfig]:
    mod = importlib.import_module(spec_module)
    if not hasattr(mod, "SPEC"):
        raise AttributeError(f"{spec_module} must define a top-level SPEC: GradientFreeDesignSpec.")
    return mod.SPEC, getattr(mod, "BASE", None) or SimulationConfig()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Lumerical-native gradient-free inverse design")
    ap.add_argument("--spec", required=True,
                    help="Module with SPEC (e.g. runners.gradient_free_design.optimize_transmission)")
    ap.add_argument("--start", type=int, default=0,
                    help="Starting-point index (default 0)")
    ap.add_argument("--baseline-lambda-nm", type=float, default=None,
                    help="Pre-measured baseline λ_resonance in nm (skips the baseline run)")
    ap.add_argument("--output-root", default=None, help="Override output directory root")
    args = ap.parse_args(argv)

    spec, base = _load_spec(args.spec)
    base_lambda_m = args.baseline_lambda_nm * 1e-9 if args.baseline_lambda_nm else None
    run_gradient_free_design(
        cfg=base,
        spec=spec,
        start_idx=args.start,
        output_root=args.output_root,
        baseline_lambda_m=base_lambda_m,
    )


if __name__ == "__main__":
    main()
