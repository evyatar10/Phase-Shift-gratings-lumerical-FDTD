"""
Scatterer DEMO — top-view field maps for the four instructive cases.

Purpose: SEE what the scatterer does to the cavity's radiated field, not just
read T/Q numbers. Re-runs four configurations from the tm_scatterer_scan /
tm_hole_scan studies with the XY (top-view) 2D field monitor ON, so MATLAB can
render |E| maps at resonance side by side:

  idx 0: baseline               — no scatterer (reference field map)
  idx 1: CONSTRUCTIVE pillar    — r=100 nm SiN pair @ (x=0.810 um, y=+/-1 um):
                                  the measured BEST net case (T 0.7995 -> 0.8015)
  idx 2: DESTRUCTIVE pillar     — r=200 nm SiN pair @ (x=1.620 um, y=+/-1 um):
                                  the measured WORST case (T -> 0.7342)
  idx 3: in-core hole           — r=100 nm SiO2 hole @ x=0 (inside the cavity)

Same anchored TM device / window / domain as tm_scatterer_scan (identical
numerics, its own baseline map). 2D monitors only (XY/YZ/XZ profiles) — modest
memory, NOT the >100 GB 3D-volume class. 51 monitor frequency points across the
30 nm band put a sample within ~0.3 nm of the resonance.

Dispatch (4 tasks; fits the QOS submit cap alongside a draining array):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_demo

Output -> results/tm_scatterer_demo/results/ (fields make these .mat larger,
~50-100 MB each). View with matlab_plotting/plot_field_poynting.m or the
study-specific demo figure script.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BASE = build_base()                      # identical device/window/domain as the scan
BASE.monitors.record_2d_fields = True    # XY top view (+ YZ/XZ) at 51 band points


SPEC = SweepSpec(
    scatterer_radius_nm = [0.0,    100.0,  200.0,  100.0],
    scatterer_x_nm      = [0.0,    810.0,  1620.0, 0.0],
    scatterer_y_nm      = [1000.0, 1000.0, 1000.0, 0.0],
    scatterer_index     = [1.97,   1.97,   1.97,   1.444],
    mode  = "zipped",
    label = "tm_scatterer_demo",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
