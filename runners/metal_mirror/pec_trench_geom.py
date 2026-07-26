"""PEC wall at the air-trench geometry — mirror-vs-light-cone discriminator (TM).

Study dir: runners/metal_mirror/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: stage L measured the air trench winning on W800 (d-opt 1.8 um,
dT +0.0159, job 124379). Is that gain "any good mirror at that position"
(interference/recoupling) or specifically the LOW-INDEX/light-cone lever?
Discriminator: one PEC wall with its FRONT FACE at y = 1.4 um — the exact
plane of the trench's reflecting oxide->air interface — same height 2 um,
same length 84 um, mirrored +/-y. PEC ~ trench => mirror story; PEC well
below => low-index/near-field story (the stage-J 350nm film's +/-0.0015 was
height- and distance-limited, not a fair test of this).
Interpretation caveat (stated up front): s-pol TIR phase at 78.5 deg is
~148 deg vs PEC's 180 deg => a mirror-mechanism optimum sits ~0.24 um away
from the trench's; a single-point PEC deficit of up to ~half is attributable
to phase offset, so only a LARGE deficit (or a match) is decisive.

Single task (no control: W800 ctrl T 0.8851 measured at IDENTICAL numerics +
sidecar in job 124379, results_from_athena/air_trench_dscan/).
Numerics: stage-H/J/L exactly (box y = 16 um, 1501 pts / 20 nm window, opt
mesh, ff base, stage-H sidecar). Physics line (CLAUDE.md section 4): TM h350,
pitch 516.83, corr 400, W800, N = 80/side; target resonance 1558.6 nm,
window 1548.5-1568.5 nm. NOTE: side FF monitor (6.75 um) is BEHIND the wall
=> shadowed; ports (T/loss/lambda/Q) + top monitor are the instruments.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.pec_trench_geom --max-concurrent=3
Output -> results/pec_trench_geom/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

MIRROR_MATERIAL = "PEC (Perfect Electrical Conductor)"
FACE_Y_NM       = 1400.0          # trench inner-edge plane (d 1800 - w/2 800)
WALL_THICK_NM   = 200.0           # moot for PEC; center sits at FACE + t/2
WALL_LEN_UM     = 84.0            # = trench length
WALL_HEIGHT_NM  = 2000.0          # = trench height

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

# Stage-H sidecar (job 123561) — identical box/window/device numerics.
LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = WALL_HEIGHT_NM * 1e-9

CENTER_Y_NM = FACE_Y_NM + 0.5 * WALL_THICK_NM
assert FACE_Y_NM >= _common.TOOTH_EDGE_NM, "wall face inside the teeth"
assert CENTER_Y_NM + 0.5 * WALL_THICK_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0

SPEC = SweepSpec(
    scatterer_shape     = ["rect"],
    scatterer_x_span_um = [WALL_LEN_UM],
    scatterer_y_span_nm = [WALL_THICK_NM],
    scatterer_y_nm      = [CENTER_Y_NM],
    scatterer_material  = [MIRROR_MATERIAL],
    mode  = "zipped",
    label = "pec_trench_geom",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"PEC wall, front face y={FACE_Y_NM} nm (= trench interface), "
          f"L={WALL_LEN_UM} um x t={WALL_THICK_NM} nm x h={WALL_HEIGHT_NM:.0f} nm, mirrored +/-y")
