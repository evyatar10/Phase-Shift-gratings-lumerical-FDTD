"""
Gradient-free study (TM): maximize peak T(λ) of the pi-shift Bragg cavity using
Lumerical's built-in Particle Swarm Optimization.

TM counterpart of optimize_transmission.py. For TM the tooth-shift degree of
freedom does NOT help, so this study frees ONLY 3 parameters:
    DW1, DW2  — corrugation depth (apodization) of the two innermost teeth
    cavity_width
The shift slots are pinned to 0 (zero-width bounds → PSO never moves them), so
the teeth stay in their regular, unshifted position. The 5-long parameter vector
[dw1, dw2, 0, 0, cavity] only exists because the shared .lsf parametric geometry
script (also used by the TE path) reads two shift slots; feeding them constant 0
is identical to a literal 3-element search and avoids touching shared code.

Pitch is recalibrated to 518.3 nm so TM resonates near 1571 nm (the TE
wavelength); the baseline scan is centered there with a wide window so
measure_baseline reliably catches the TM peak.

Invocation:
  Athena (single-GPU job):
    bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission_tm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.gradient_free_design.gradient_free_design import GradientFreeDesignSpec
from runners.optimization_common import make_optimization_base


BASE = make_optimization_base(n_periods=80)

# ── TM polarization + calibrated pitch ──────────────────────────────────────
BASE.source.polarization          = "TM"
BASE.grating.pitch_m              = 518.3e-9      # calibrated TM pitch (calibrate_neff)

# IT11-calibrated CONSTANT indices for TM. The simulation_config default that
# make_optimization_base inherits (1.977/1.44) is WRONG for TM; all other TM
# work uses these (runners/tm/_tm_vs_te_common.py). Pitch 518.3 was calibrated
# against THIS index, so with it TM resonates near ~1571 nm.
BASE.material.use_constant_materials = True
BASE.material.n_core_const        = 1.9963
BASE.material.n_clad_const        = 1.444

# Baseline scan: wide window straddling both the ~1562 nm (wrong-index) and
# ~1571 nm (correct-index) candidates so measure_baseline can't miss the peak.
BASE.spectral.center_wavelength_m = 1.567e-6      # 1567 nm
BASE.spectral.scan_width_nm       = 34.0          # baseline scan [1550, 1584] nm
BASE.material.n_eff_guess         = BASE.spectral.center_wavelength_m / (2 * BASE.grating.pitch_m)


N_FREE = 2
# Regular-grating seed: full corrugation depth, NO shift, average cavity width.
INITIAL_P = [300.0, 300.0, 0.0, 0.0, 800.0]

SPEC = GradientFreeDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],   # DW1, DW2 (apodization)
    shift_bounds_nm    = [(0.0, 0.0),   (0.0, 0.0)],       # shift is NOT a parameter for TM
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # PSO budget: 3 free dims → smaller swarm than the TE run converges.
    # pop=12 × (gens=10 + 1 init) ≈ 132 evals. Incremental save after each gen
    # so a walltime-cut run still yields the global-best particle to date.
    algorithm          = "Particle Swarm",
    population_size    = 12,
    max_generations    = 10,
    tolerance          = 1e-4,
    n_concurrent       = 1,         # single-GPU sequential

    fom_window_nm      = 16.0,      # wider than TE: tolerates resonance drift vs cavity_width
    fom_n_points       = 401,       # finer: TM Q~4900 → FWHM~0.3 nm needs ~10 pts/FWHM
    mesh_override_dxyz_nm = 0,      # device-wide 50 nm mesh (fine override intractable at N=80)

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    # Rebuild the FULL device each particle via the production builder. The
    # parametric .fsp path (default) produces a DEAD TM device (job 97162:
    # modal T≈0.0008 for every particle incl. the regular-grating baseline,
    # which the normal builder transmits at ~0.95). This backend is immune to
    # that divergence — same path as measure_baseline.
    rebuild_per_particle = True,
    label              = "transmission_gf_tm",
)


if __name__ == "__main__":
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design
    print(SPEC.describe())
    print()
    run_gradient_free_design(cfg=BASE, spec=SPEC, start_idx=0)
