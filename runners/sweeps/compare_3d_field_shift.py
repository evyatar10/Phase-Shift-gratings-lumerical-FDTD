"""
Sweep: 3D-field comparison of two devices (innermost_tooth_shift = 0 vs 100 nm).

This is the PARALLEL-array half of the chained two-job workflow that replaces
the sequential runners/single/compare_3d_field_default_vs_shift.py on Athena:

    Job 1  prelim sim                           (single GPU, ~5 min)
           RUN_SCRIPT=compare_3d_field_prelim
           → writes λ_res to LOCKED_LAMBDA_FILE

    Job 2  this sweep, SLURM array of 2         (1 GPU per task, parallel)
           --dependency=afterok:Job1
           → each task reads LOCKED_LAMBDA_FILE, sets center_wavelength_m,
             runs locked-wavelength 3D + far-field.

deploy_athena.sh recognizes the PRELIM_RUN_SCRIPT / LOCKED_LAMBDA_FILE
module-level constants below and submits the two jobs in the right order.
build_sweep_list.py forwards them as SWEEP_META lines for the deploy script;
athena_run_one.py picks up LOCKED_LAMBDA_FILE at task start.

Run locally (sequential, no SLURM, no chaining):
    python -m runners.sweeps.compare_3d_field_shift

  → defers to runners/single/compare_3d_field_default_vs_shift.py, which does
    prelim + both big sims back-to-back in the same Python process. The
    sequential variant is also what Zeus uses (no job-array support there).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from simulation_config import SimulationConfig


# ── Spec-module hooks read by build_sweep_list.py / deploy_athena.sh ──────────
# Presence of PRELIM_RUN_SCRIPT triggers the prelim → array chain on Athena.
# Modules without these constants behave exactly as before (single array,
# no prerequisite).
PRELIM_RUN_SCRIPT  = "compare_3d_field_prelim"
LOCKED_LAMBDA_FILE = "/work/results/compare_3d_lambda_res.json"


# ── Base config shared by both array tasks ────────────────────────────────────
# All non-swept settings live here so the SPEC stays minimal. The base is read
# by athena_run_one._run_kind_spec via getattr(module, 'BASE', None) and
# passed into SPEC.expand(base=BASE). spectral.center_wavelength_m is then
# overwritten per-task at runtime from LOCKED_LAMBDA_FILE.
BASE = SimulationConfig()
BASE.monitors.record_3d_fields    = True
BASE.monitors.record_2d_fields    = False
BASE.monitors.n_3d_freq_points    = 1      # locked-wavelength snapshot — keeps the 3D array small
BASE.farfield.enabled             = True   # widens the simulation domain to 5 λ
BASE.grating.cavity_width_option  = "narrow"
BASE.spectral.scan_width_nm       = 6.0    # narrow band around λ_res


SPEC = SweepSpec(
    innermost_tooth_shift_nm = [0, 100],
    label = "compare_3d_field_shift",
)


if __name__ == "__main__":
    # Local entrypoint hands off to the sequential single-script variant.
    # Same end result, no SLURM coordination needed off-cluster.
    from runners.single.compare_3d_field_default_vs_shift import (
        run_compare_3d_field_default_vs_shift,
    )
    run_compare_3d_field_default_vs_shift()
