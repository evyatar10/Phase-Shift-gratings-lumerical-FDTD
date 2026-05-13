"""
Inverse-design study: maximize peak transmission of the pi-shift Bragg cavity
by tuning the two innermost teeth (DW + shift on each) and the cavity width.

Invocation:
  Local (one driver, sequential):
    python -m runners.inverse_design.inverse_design --spec runners.inverse_design.optimize_transmission
  Athena (SLURM, parallel-ready via concurrent_adjoint_solves=True):
    bash athena/deploy_athena.sh --inverse-design=runners.inverse_design.optimize_transmission
  DGX:
    bash dgx/deploy_dgx.sh --inverse-design=runners.inverse_design.optimize_transmission
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.inverse_design.inverse_design import InverseDesignSpec, regular_grating_start
from simulation_config import SimulationConfig


# ── Base simulation config (matches the inverse-design plan) ─────────────────

BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.grating.lengthen_cavity     = True             # NOT optimizing cavity length;
                                                    # this absorbs Σ(shifts) so total
                                                    # device length stays constant.
BASE.mesh.simulation_mode        = "optimization"   # device-wide dx=50 nm; the
                                                    # freed region gets a finer
                                                    # 10 nm override (Phase-2 fix
                                                    # #2) wired in inverse_design.
BASE.spectral.scan_width_nm      = 10.0             # full bandgap window; FOM
                                                    # weight (σ=1 nm Gaussian)
                                                    # restricts the integral.
# Apodization stays at its default (off); freed inner teeth carry their own DW.

# ── Optional field profile monitors (default OFF for inverse design) ────────
# Optimization only needs T(λ) at Port_2 (PortTransmission FOM); profile
# monitors are pure overhead during the 60+ FDTD runs.
# Flip any of these to True if you want field profiles recorded during the run
# (e.g. for post-mortem visualization of the optimized geometry's mode shape).
BASE.monitors.record_2d_fields = False   # XY top, YZ cross, XZ side profiles
BASE.monitors.record_3d_fields = False   # full 3D field volume
BASE.farfield.enabled          = False   # side + top far-field monitors


# ── Inverse-design spec ──────────────────────────────────────────────────────
# Single deterministic start at the regular-grating geometry. Peak T at this
# start must be reproduced by the optimizer's first iteration; any drop is a
# wiring bug. Multi-start LHS sampling is available via spec.get_starts() but
# disabled here since the regular grating is a known-good convergent baseline.

N_FREE = 2

# Empirically-informed starting point. The regular grating ([300,300,0,0,800])
# is a known LOCAL plateau of the FOM landscape — gradient nearly zero, L-BFGS-B
# stalls. Per the user's prior sweep work (innermost_shift, apod_and_shift),
# improvement comes from APODIZING the inner teeth (smaller DW) and adding
# a TOOTH SHIFT. Starting near a known-good empirical region gives the
# optimizer a meaningful initial gradient and avoids the regular-grating
# saddle.
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]   # apodized + slightly shifted

SPEC = InverseDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    # half_pitch = 250 nm, so 200 nm shift leaves 50 nm narrow-tooth min.
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # PHASE-3: lumopt adjoint now works (vec_error 0.144 in job 79505 after
    # the 4-fix stack). 8 iterations is enough to verify it actually steps;
    # bump to 16+ once we see real improvement.
    max_iter           = 8,
    optimizer_method   = "L-BFGS-B",
    optimizer_pgtol    = 1e-6,
    optimizer_ftol     = 1e-6,
    # Force first L-BFGS-B step to be 25% of bound range. Without this the
    # raw gradient (~1e-4 in scaled [0,1] space) translates to sub-Angstrom
    # physical steps, the line search rejects all candidates, and the
    # optimizer terminates after one iteration with no real movement
    # (smoke job 80222 saw exactly this: FOM 0.250 → 0.250 in 1 iter).
    scale_initial_gradient_to = 0.25,

    # FOM: Gaussian-weighted T over the 10 nm bandgap window.
    # σ chosen against MEASURED spectral FWHM_T ≈ 1.05 nm (production N=80,
    # Q ≈ 1485, source: result_N80_W800.mat). With σ=2 nm, FWHM_Gaussian =
    # 2.355·σ = 4.7 nm = 4.5× FWHM_T → robust to 2-3 nm resonance drift.
    # σ=1 nm collapses the gradient signal at 2 nm drift (weight → exp(-2) ≈ 0.14).
    # NB: this is the spectral FWHM of T(λ), NOT the spatial energy-envelope
    # FWHM (~7.7 µm), which is irrelevant for a wavelength-axis weight.
    fom_window_nm        = 10.0,
    fom_n_points         = 201,
    fom_weight_sigma_nm  = 2.0,

    # Freed-region mesh override at 15 nm (user-requested 2026-05-13).
    # Tighter than the prior 25 nm default; gives more accurate d_eps for
    # cavity-shape and tooth-edge perturbations. param_dx_nm (50 nm) is
    # comfortably > 3× cell, so finite-difference perturbations stay
    # well-resolved on the discretized grid.
    mesh_override_dxyz_nm= 15,
    param_dx_nm          = 50.0,

    # Serial adjoint solves (concurrent=True needs >1 license token at
    # once → fails with concurrent jobs eating the FlexLM pool).
    use_concurrent_adjoint_solves = False,
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "transmission",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design
    print(SPEC.describe())
    print()
    run_inverse_design(cfg=BASE, spec=SPEC, start_idx=0)
