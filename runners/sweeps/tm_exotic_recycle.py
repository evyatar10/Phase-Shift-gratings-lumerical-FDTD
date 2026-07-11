"""
EXOTIC innermost-tooth-PAIR recycler — quick empirical TM test (USER-DIRECTED).

Reshape ONLY the innermost tooth on each side of the cavity (one left, one right;
n_shaped = 1, x-mirrored, y-symmetric) with an *interesting* shape, on the N=80 TM
anchored device. Goal (user): a shape that cancels the Lorentzian near-grazing leak so
the interference is CONSTRUCTIVE inside the cavity (recycle inward -> less loss). Theory
says a single-tooth shape is a small lever; this is the empirical test of the best
candidate shapes anyway.

Best-candidate shapes (innermost pair only; plain single-device grating base, so
kappa / spatial mode width are essentially untouched — the user's constraint):
  row 0  rect        control (identical numerics)
  row 1  step        blocky two-level tooth, taller on the cavity side (secondary
                     edge tilted inward — a blazed recycler; PHASE lever)
  row 2  notch       slot in the top-center -> two sub-peaks (localized double-
                     lattice = a second scattering edge; destructive-outside PHASE lever)
  row 3  wedge_cav   triangular face tilted TOWARD the cavity (redirect the leak inward)

TM, N=80/side. Narrow window (one shaped tooth -> tiny blue-shift); resonance finder
locates each peak; loss compared at each device's own resonance (never argmax T). Opt
mesh (quick); any survivor gets accurate + half-cell jitter before a claim. Watch for
T>1 / negative loss (box overflow) — unlikely with only 1 shaped tooth.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_exotic_recycle
Output -> results/tm_exotic_recycle/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8       # converged y (job 116854)
BOX_Z_MULT = 5.42    # converged z (job 116870): z = 8.8 um

BASE = build_base()                    # N=80, TM (source.polarization='TM'), anchored geometry
BASE.scatterer.enabled = False         # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
# One shaped tooth per side -> only a tiny blue-shift; narrow window around 1558.6 nm.
BASE.spectral.center_wavelength_m = 1.5580e-6
BASE.spectral.scan_width_nm = 50.0
BASE.spectral.n_wl_points = 4001

_shapes = ["rect", "step", "notch", "wedge_cav"]
_nsh    = [1,      1,      1,       1]          # innermost tooth PAIR only

assert len(_shapes) == len(_nsh) == 4
assert len(set(zip(_shapes, _nsh))) == 4       # unique file tags

SPEC = SweepSpec(
    inner_shape    = _shapes,
    n_shaped_teeth = _nsh,
    mode  = "zipped",
    label = "tm_exotic_recycle",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
