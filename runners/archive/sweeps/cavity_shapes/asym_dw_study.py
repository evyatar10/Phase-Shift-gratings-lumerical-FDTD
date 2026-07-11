"""
ANTI-RADIATOR study — antisymmetric inner-tooth depth detuning (exotic center idea).

Concept (new — not apodization, not tooth shift, not shape): give the left arm's
tooth d a corrugation depth corr+delta and the right arm's tooth d corr-delta.
This ODD perturbation on the EVEN cavity mode:
  * does not shift lambda_res or kappa to first order (left/right cancel) ->
    the spatial mode width is untouched (fwhm_m recorded to verify);
  * DOES radiate, with an amplitude set by delta and a phase set by the tooth
    position x_d (through e^{+-i k_clad x_d} toward the +-x lobes). Because the
    odd source enters the forward and backward lobes with the SAME magnitude
    |sin(k x_d)|, one (delta, d) pair can destructively interfere with the
    defect's own radiation in BOTH near-axial TM lobes at once — unlike any
    external scatterer or wall-offset trick.
The defect radiation's amplitude/phase is unknown a priori (paper_8's "needs
FDTD" part), so this is a 2-knob empirical scan: tooth index d = 1..3 x delta =
{15, 30, 60} nm. If ANY cell shows loss below control beyond the noise floor,
the phase knob works and a finer (delta, d) map + accurate-mesh confirm follows.
delta -> -delta is the x-mirror of the same device (identical T), so only
positive deltas are scanned.

Rows (zipped, 10 tasks): control + d in {1,2,3} x delta in {15,30,60} nm.
Anchored TM device, converged box 6.8/8.8 um, optimization mesh, window
1558.5 / 40 nm / 3001 pts. y-symmetry stays ON (widths remain y-symmetric).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.asym_dw_study
Output -> results/asym_dw_study/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8       # converged y (job 116854)
BOX_Z_MULT = 5.42    # converged z (job 116870): z = 8.8 um

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001


def _delta_list(d, delta_nm):
    """Zero-padded per-tooth delta list with delta at tooth index d (1-based)."""
    v = [0.0] * d
    v[d - 1] = float(delta_nm)
    return v


_rows = [None]                                    # control (symmetric device)
for _d in (1, 2, 3):
    for _delta in (15.0, 30.0, 60.0):
        _rows.append(_delta_list(_d, _delta))

assert len(_rows) == 10
_keys = {tuple(r) if r else () for r in _rows}
assert len(_keys) == 10, "duplicate (d, delta) row -> tag collision"

SPEC = SweepSpec(
    asym_dw_delta_nm = _rows,
    mode  = "zipped",
    label = "asym_dw_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
