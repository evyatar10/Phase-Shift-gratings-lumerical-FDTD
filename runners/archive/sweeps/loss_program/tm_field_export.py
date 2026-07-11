"""
FIELD EXPORT for the DERIVED-SHAPE derivation (user-selected route, 2026-07-06).

Purpose: the boundary-perturbation ("golden shape") derivation needs the
complex resonant field E(x, y) at core mid-height (z=0) near the inner-region
sidewalls of the CURRENT BEST DEVICE — the stack (cavity 1050 + gap pair
[+20,+20] + see-saw 1040/980). Only 1D envelopes were stored by the loss
sweeps. This 2-row study records the 2D XY field at accurate mesh.

The derivation (python_tools/derive_boundary_profile.py) then:
  1. reads E(x,y) at z=0, samples E_z along the sidewall contour y = ±W(x)/2,
  2. computes the residual in-cone amplitude A0(kx) (k-space diagnostic method),
  3. solves the constrained least-squares for the sidewall perturbation
     delta(x) that cancels A0 in-cone while keeping the Bragg (2*beta)
     component (kappa / mode width) and the resonance unchanged,
  4. reports whether delta(x) projects onto the already-used knobs
     (shift = 1st moment, see-saw = 2nd) or has ORTHOGONAL content — the
     decision variable for whether a "golden shape" exists off the frontier.

Rows (zipped, 2 tasks, accurate mesh, converged box y=6.8/z-mult 5.42,
window 1558.5 / 40 nm / 3001):
   0  the stack (W1050 + pair[+20,+20] + see-saw 1040/980)  — the target
   1  W800 baseline                                          — pipeline
      validation against the existing k-space dataset (radiation_kspace_diag)

Monitors: 2D fields ON (XY top view at z=0), x-span 24 um (inner region ±12 um
covers the defect lobe = the derivation domain; 77% of side radiation
originates inside ±12 um per polarimetry). record_3d/farfield OFF. Accurate
mesh -> E_res ~ 5 MB/row in the .mat: fine for the link.

Dispatch (queue must be EMPTY of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_field_export
Output -> results/tm_field_export/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

# 2D XY field at z=0 over the inner +-12 um — the derivation domain.
BASE.monitors.record_2d_fields = True
BASE.monitors.field_2d_x_span_m = 24e-6
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled = False

_n = 2
SPEC = SweepSpec(
    cavity_width_nm           = [1050.0, None],
    inner_shift_list_nm       = [[20.0, 20.0], None],
    n_free_inner_teeth        = [2, 1],
    width_narrow_per_tooth_nm = [[600.0, 600.0], None],
    width_wide_per_tooth_nm   = [[1040.0, 980.0], None],
    simulation_mode           = ["accurate"] * _n,
    mode  = "zipped",
    label = "tm_field_export",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
