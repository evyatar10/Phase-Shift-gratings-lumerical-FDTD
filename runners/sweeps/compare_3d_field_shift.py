"""
Sweep: 3D-field comparison of two devices (innermost_tooth_shift = 0 vs 100 nm).

Two deployment modes, toggled by USE_PRELIM below:

  USE_PRELIM = True  (default)
    Job 1  prelim array of 2 — defined by PRELIM_SPEC / PRELIM_BASE below.
           Finds λ_res per shift option, writes per-task JSON sidecars.
    Job 2  this sweep, SLURM array of 2  (--dependency=afterok:Job1)
           reads each sidecar and runs locked-λ 3D + far-field.
    Submit ONCE — deploy_athena.sh chains the two arrays automatically.

  USE_PRELIM = False
    Skip the prelim; λ_res values are hardcoded from a prior run.
    Only valid when the device geometry and materials are unchanged.
    Update KNOWN_RESONANCES_NM whenever you re-characterise the device.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from simulation_config import SimulationConfig


# ══════════════════════════════════════════════════════════════════════════════
# Toggle here
# ══════════════════════════════════════════════════════════════════════════════
USE_PRELIM = False

# Used only when USE_PRELIM = False. Update if geometry or materials change.
SHIFTS_NM           = [0,        100      ]
KNOWN_RESONANCES_NM = [1560.103, 1560.480 ]   # from full 3D run 2026-04-29
# ══════════════════════════════════════════════════════════════════════════════


# ── Main sim config ───────────────────────────────────────────────────────────
BASE = SimulationConfig()
BASE.monitors.record_3d_fields    = True
BASE.monitors.record_2d_fields    = False
BASE.monitors.n_3d_freq_points    = 1       # single-bin snapshot at λ_res
BASE.farfield.enabled             = True    # 5 λ domain for far-field capture
BASE.grating.cavity_width_option  = "narrow"
BASE.spectral.scan_width_nm       = 10.0    # 10 nm band centered on λ_res


if USE_PRELIM:
    # Prelim config: fast resonance scan, no 3D/2D/far-field, narrow domain.
    # Gated here so build_sweep_list only detects PRELIM_SPEC when USE_PRELIM=True.
    PRELIM_BASE = SimulationConfig()
    PRELIM_BASE.monitors.record_3d_fields    = False
    PRELIM_BASE.monitors.record_2d_fields    = False
    PRELIM_BASE.farfield.enabled             = False
    PRELIM_BASE.grating.cavity_width_option  = "narrow"
    PRELIM_BASE.spectral.center_wavelength_m = 1560e-9
    PRELIM_BASE.spectral.scan_width_nm       = 20.0

    PRELIM_SPEC = SweepSpec(
        innermost_tooth_shift_nm = SHIFTS_NM,
        label                    = "compare_3d_field_prelim",
    )

    # {idx} expands to SLURM_ARRAY_TASK_ID so task k reads the sidecar written
    # by prelim task k.
    LOCKED_LAMBDA_FILE = "/work/results/compare_3d_lambda_res_{idx}.json"

    SPEC = SweepSpec(
        innermost_tooth_shift_nm = SHIFTS_NM,
        label                    = "compare_3d_field_shift",
    )
else:
    # No prelim. center_wavelength_nm is zipped with shift so each task gets
    # the exact λ_res from the last full run.
    SPEC = SweepSpec(
        innermost_tooth_shift_nm = SHIFTS_NM,
        center_wavelength_nm     = KNOWN_RESONANCES_NM,
        mode                     = "zipped",
        label                    = "compare_3d_field_shift",
    )


if __name__ == "__main__":
    from runners.sweeps.sweep_spec import run_sweep_spec
    run_sweep_spec(SPEC, target="local", base=BASE)
