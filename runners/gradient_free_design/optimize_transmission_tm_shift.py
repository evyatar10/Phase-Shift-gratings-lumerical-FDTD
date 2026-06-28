"""
Gradient-free study (TM, 5-parameter) — EXPLORATORY comparison run.

Same TM physics as optimize_transmission_tm.py (pitch 518.3 nm, constant
n_core=1.9963 / n_clad=1.444), but this run FREES THE TOOTH SHIFT back in: it
optimizes all 5 parameters
    DW1, DW2     — corrugation depth (apodization) of the two innermost teeth
    SHIFT1, SHIFT2 — longitudinal shift of those teeth
    cavity_width
The purpose is purely to check whether re-adding the shift DOF buys any TM peak-T
over the 3-param result (where shift was found not to help). Separate `label`
(transmission_gf_tm_shift) → separate results dir, so the existing 3-param TM
results are untouched.

Invocation:
  Athena (single-GPU job):
    bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission_tm_shift
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.gradient_free_design.gradient_free_design import GradientFreeDesignSpec
from runners.optimization_common import make_optimization_base


BASE = make_optimization_base(n_periods=80)

# ── TM polarization + calibrated pitch (identical to optimize_transmission_tm) ─
BASE.source.polarization          = "TM"
BASE.grating.pitch_m              = 516.14e-9     # calibrated TM pitch (FDTD bracket, n 1.97)

# CONSTANT indices for TM. The simulation_config default that make_optimization_base
# inherits is WRONG for TM; all other TM work uses these (runners/tm/_tm_vs_te_common.py).
# Pitch 516.14 was calibrated against n_core 1.97 (user 2026-06-28; was 1.9963/518.3),
# so with it TM resonates near ~1558.7 nm.
BASE.material.use_constant_materials = True
BASE.material.n_core_const        = 1.97
BASE.material.n_clad_const        = 1.444

# Baseline scan: wide window straddling the candidate resonances so
# measure_baseline can't miss the TM peak.
BASE.spectral.center_wavelength_m = 1.567e-6      # 1567 nm
BASE.spectral.scan_width_nm       = 34.0          # baseline scan [1550, 1584] nm
BASE.material.n_eff_guess         = BASE.spectral.center_wavelength_m / (2 * BASE.grating.pitch_m)


N_FREE = 2
# SMART seed: start from the known 3-param TM optimum (optimize_transmission_tm,
# label transmission_gf_tm) — DW1≈95.2, DW2≈102.0, cavity≈854.3 nm, shift=0,
# which reached true_peak_T≈0.956. Shift starts at 0 but is now FREE, so the
# swarm tests whether ADDING shift improves on that already-good design rather
# than re-discovering it from the generic regular-grating point. The other 14
# particles randomize across the full 5-D box, so global exploration is intact.
INITIAL_P = [95.18, 102.0, 0.0, 0.0, 854.30]

SPEC = GradientFreeDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],   # DW1, DW2 (apodization)
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],    # SHIFT FREED (vs the 3-param run)
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # PSO budget: 5 free dims → larger swarm than the 3-param TM run.
    # pop=15 × (gens=10 + 1 init) ≈ 165 evals. Incremental save after each gen
    # so a walltime-cut run still yields the global-best particle to date.
    algorithm          = "Particle Swarm",
    population_size    = 15,
    max_generations    = 10,
    tolerance          = 1e-4,
    n_concurrent       = 1,         # single-GPU sequential

    fom_window_nm      = 16.0,      # wider than TE: tolerates resonance drift vs cavity_width
    fom_n_points       = 401,       # finer: TM Q~4900 → FWHM~0.3 nm needs ~10 pts/FWHM
    mesh_override_dxyz_nm = 0,      # device-wide 50 nm mesh (fine override intractable at N=80)

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    # Rebuild the FULL device each particle via the production builder. The
    # parametric .fsp path (default) produces a DEAD TM device (job 97162);
    # this backend is immune to that divergence — same path as measure_baseline.
    rebuild_per_particle = True,
    label              = "transmission_gf_tm_shift",   # SEPARATE results dir
)


if __name__ == "__main__":
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design
    print(SPEC.describe())
    print()
    run_gradient_free_design(cfg=BASE, spec=SPEC, start_idx=0)
